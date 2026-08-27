#!/usr/bin/env python3
"""Test calibration impact on Week 24"""
import collections
import os
import sys

if os.getenv("RUN_LIVE_TESTS") != "1":
    print("SKIP: set RUN_LIVE_TESTS=1 to run the Snowflake-backed calibration test.")
    sys.exit(0)

import app

week, year = 24, 2025
ground_truth_idxs = {9, 27, 28, 57, 75, 102, 153, 160, 162}

# Clear caches
app.load_data_for_week.clear()
app.analyze_emotions_cached.clear()

df = app.load_data_for_week(week, year, show_progress=False)
if df.empty:
    print("ERROR: live calibration requested, but Week 24 data is unavailable.", file=sys.stderr)
    sys.exit(1)
df_em = app.analyze_emotions_cached(df, model_id='SamLowe/roberta-base-go_emotions')

print("="*70)
print("WEEK 24 CALIBRATION TEST - POST-4-POINT IMPLEMENTATION")
print("="*70)

# Run Tier-1 detection
flagged = []
reasons = collections.Counter()

for i, row in df_em.iterrows():
    orig = df.iloc[i]
    if app.is_staff_or_admin_comment(orig.get('commenter_name', ''), orig.get('commenter_email', '')):
        continue
    
    detail = app.compute_severe_candidate_details(row, str(orig.get('content', '')))
    if detail.get('candidate'):
        idx = i + 1  # 1-indexed
        flagged.append(idx)
        reasons.update(detail.get('reason_flags', []))

# Compare with ground truth
hits = sorted(ground_truth_idxs.intersection(flagged))
misses = sorted(ground_truth_idxs - set(flagged))

print(f"\nTier-1 Candidates: {len(flagged)}")
print(f"Ground-Truth Hits: {len(hits)}/9")
print(f"Hit Indices: {hits}")
print(f"Missed Indices: {misses}")
print(f"\nTop Reason Flags:")
for flag, count in reasons.most_common(12):
    print(f"  {flag}: {count}")

print("\n" + "="*70)
print("CANDIDATE DETAILS")
print("="*70)
for idx in sorted(flagged)[:20]:
    i = idx - 1
    orig = df.iloc[i]
    row = df_em.iloc[i]
    detail = app.compute_severe_candidate_details(row, str(orig.get('content', '')))
    print(f"\n[{idx}] {orig.get('commenter_name', 'UNKNOWN')}")
    print(f"    Flags: {detail.get('reason_flags', [])}")
    content_preview = str(orig.get('content', ''))[:80]
    print(f"    Content: {content_preview}...")
