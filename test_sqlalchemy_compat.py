#!/usr/bin/env python3
"""Credential-free Snowflake SQLAlchemy compatibility smoke test."""
import importlib.metadata
import sys


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main():
    engine = None
    try:
        import snowflake.sqlalchemy  # noqa: F401
        from sqlalchemy import create_engine, text

        engine = create_engine(
            "snowflake://compat_user@compat_account/compat_db/compat_schema?warehouse=compat_wh"
        )
        assert engine.dialect.name == "snowflake"
        compiled = str(text("SELECT 1").compile(dialect=engine.dialect))
        assert compiled == "SELECT 1"
        print("[OK] Snowflake dialect imported, constructed, and compiled SELECT 1 without connecting")
        return 0
    except Exception as error:
        print(f"[ERROR] Snowflake SQLAlchemy compatibility check failed: {error}", file=sys.stderr)
        for package in ("SQLAlchemy", "snowflake-sqlalchemy", "snowflake-connector-python"):
            print(f"        {package}={package_version(package)}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
