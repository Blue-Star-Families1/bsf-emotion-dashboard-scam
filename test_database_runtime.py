#!/usr/bin/env python3
"""Offline regression tests for weekly Snowflake load result semantics."""
import sys
from pathlib import Path

import pandas as pd
from snowflake.connector.errors import ProgrammingError

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import app


class FakeEngineLoader:
    def __init__(self):
        self.clear_count = 0

    def __call__(self):
        return object(), True

    def clear(self):
        self.clear_count += 1


def assert_raises_load_error(action, label):
    try:
        action()
    except app.WeeklyDataLoadError:
        print(f"[OK] {label}")
        return
    raise AssertionError(f"Expected WeeklyDataLoadError: {label}")


def run_load_case(engine_loader, read_sql):
    original_engine_loader = app.get_snowflake_engine
    original_read_sql = app.pd.read_sql
    app.get_snowflake_engine = engine_loader
    app.pd.read_sql = read_sql
    app.load_data_for_week.clear()
    try:
        return app.load_data_for_week(24, 2025, show_progress=False)
    finally:
        app.load_data_for_week.clear()
        app.get_snowflake_engine = original_engine_loader
        app.pd.read_sql = original_read_sql


def main():
    try:
        assert_raises_load_error(
            lambda: run_load_case(lambda: (None, False), lambda *_args, **_kwargs: None),
            "engine unavailability is not reported as empty data",
        )

        def failed_query(*_args, **_kwargs):
            raise RuntimeError("offline query failure")

        assert_raises_load_error(
            lambda: run_load_case(FakeEngineLoader(), failed_query),
            "query failure is not reported as empty data",
        )

        empty_result = run_load_case(
            FakeEngineLoader(),
            lambda *_args, **_kwargs: pd.DataFrame(columns=app.WEEKLY_DATA_COLUMNS),
        )
        assert empty_result.empty
        assert list(empty_result.columns) == app.WEEKLY_DATA_COLUMNS
        print("[OK] successful zero-row query remains a standardized empty result")

        populated_result = run_load_case(
            FakeEngineLoader(),
            lambda *_args, **_kwargs: pd.DataFrame([{
                "activity_type": "Comment",
                "activity_date": "2025-06-12",
                "content": "<p>Hello &amp; welcome</p>",
                "community_name": "Test",
                "commenter_name": "Member",
                "commenter_email": "member@example.com",
            }]),
        )
        assert len(populated_result) == 1
        assert populated_result.iloc[0]["content"] == "Hello & welcome"
        print("[OK] successful populated query still cleans and returns data")

        retry_loader = FakeEngineLoader()
        attempts = {"count": 0}

        def expired_then_success(*_args, **_kwargs):
            attempts["count"] += 1
            if attempts["count"] == 1:
                error = RuntimeError("wrapped connector error")
                error.orig = ProgrammingError("Authentication token has expired")
                raise error
            return pd.DataFrame(columns=app.WEEKLY_DATA_COLUMNS)

        retry_result = run_load_case(retry_loader, expired_then_success)
        assert retry_result.empty
        assert attempts["count"] == 2
        assert retry_loader.clear_count == 1
        print("[OK] expired authentication token clears the engine and retries once")

        original_week_loader = app.load_data_for_week
        app.load_data_for_week = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            app.WeeklyDataLoadError("rolling-window failure")
        )
        try:
            assert_raises_load_error(
                lambda: app.load_scam_detection_window(24, 2025),
                "rolling scam window propagates database failures",
            )
        finally:
            app.load_data_for_week = original_week_loader

        print("RESULT: Weekly database runtime semantics validated successfully")
        return 0
    except Exception as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
