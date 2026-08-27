#!/usr/bin/env python3
"""Week 24 calibration validation - measure impact of 4-point improvements"""
import collections
import os
import sys

if os.getenv("RUN_LIVE_TESTS") != "1":
    print("SKIP: set RUN_LIVE_TESTS=1 to run the Snowflake-backed Week 24 test.")
    sys.exit(0)

import app

week, year = 24, 2025
# Verified ground-truth: only non-staff comments with clear member need/hardship
# Removed: 27, 75, 153 = Sierra Martinez (smartinez@bluestarfam.org = BSF staff, skipped by pipeline)
# Removed: 57 (new member intro), 102 (holiday greeting), 160 (gaming post),
#   162 (informational outreach), 28 (neutral/optimistic, no distress signal)
ground_truth_idxs = {9}

print("Clearing caches...")
app.load_data_for_week.clear()
app.analyze_emotions_cached.clear()

print("Loading data...")
df = app.load_data_for_week(week, year, show_progress=False)
print(f"Total rows: {len(df)}")
if df.empty:
    print("ERROR: live Week 24 test requested, but source data is unavailable.", file=sys.stderr)
    sys.exit(1)

print("Running emotion analysis...")
df_em = app.analyze_emotions_cached(df, model_id='SamLowe/roberta-base-go_emotions')

print("\n" + "="*75)
print("WEEK 24 - POST CALIBRATION ANALYSIS")
print("="*75)

# Run Tier-1 detection
flagged = []
reasons = collections.Counter()
flagged_details = {}

for i, row in df_em.iterrows():
    orig = df.iloc[i]
    if app.is_staff_or_admin_comment(orig.get('commenter_name', ''), orig.get('commenter_email', '')):
        continue
    
    detail = app.compute_severe_candidate_details(row, str(orig.get('content', '')))
    if detail.get('candidate'):
        idx = i + 1
        flagged.append(idx)
        reasons.update(detail.get('reason_flags', []))
        flagged_details[idx] = {
            'name': orig.get('commenter_name', 'UNKNOWN'),
            'content': str(orig.get('content', ''))[:100],
            'flags': detail.get('reason_flags', [])
        }

hits = sorted(ground_truth_idxs.intersection(flagged))
misses = sorted(ground_truth_idxs - set(flagged))

n_gt = len(ground_truth_idxs)
print(f"\n📊 METRICS:")
print(f"   Tier-1 Candidates: {len(flagged)}")
print(f"   Ground-Truth Hits: {len(hits)}/{n_gt} ({100*len(hits)/n_gt:.0f}%)")
print(f"   Recall: {len(hits)}/{n_gt}")
print(f"   Hit Indices: {hits}")
print(f"   Missed Indices: {misses}")

print(f"\n🏷️  TOP REASON FLAGS:")
for flag, count in reasons.most_common(12):
    print(f"   {flag}: {count}")

print(f"\n✅ HIT DETAILS (ground-truth candidates flagged):")
for idx in sorted(hits):
    detail = flagged_details[idx]
    print(f"   [{idx}] {detail['name']}")
    print(f"       Flags: {detail['flags']}")
    print(f"       Content: {detail['content']}...")

print(f"\n❌ MISS DETAILS (ground-truth candidates missed):")
for idx in sorted(misses):
    i = idx - 1
    orig = df.iloc[i]
    content_preview = str(orig.get('content', ''))[:100]
    print(f"   [{idx}] {orig.get('commenter_name', 'UNKNOWN')}")
    print(f"       Content: {content_preview}...")

print("\n" + "="*75)
