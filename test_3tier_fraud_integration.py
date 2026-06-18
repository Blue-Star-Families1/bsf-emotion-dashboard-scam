#!/usr/bin/env python3
"""Quick integration test for 3-tier fraud detection in detect_scam_concerns()"""
import app

app.load_data_for_week.clear()
df = app.load_data_for_week(24, 2025, show_progress=False)
print(f"Loaded {len(df)} Week 24 comments")

# Run detect_scam_concerns with new 3-tier pipeline
print("\nRunning detect_scam_concerns with 3-tier fraud detection...")
scam_concerns = app.detect_scam_concerns(df, silent=True)
print(f"\nTier-1 + Tier-2 + Tier-3 Results:")
print(f"  Total flagged: {len(scam_concerns)}")

if scam_concerns:
    # Check for Tier 3 verdicts
    tier3_cases = [c for c in scam_concerns if c.get('tier3_verdict')]
    print(f"  Tier-3 context gates applied: {len(tier3_cases)}")
    
    if tier3_cases:
        print(f"\n  Tier-3 verdicts breakdown:")
        from collections import Counter
        verdicts = Counter(c.get('tier3_verdict') for c in tier3_cases)
        for v, count in verdicts.most_common():
            print(f"    {v}: {count}")
    
    # Show action distribution
    print(f"\n  Recommended actions:")
    actions = Counter(c.get('recommended_action') for c in scam_concerns)
    for action, count in actions.most_common():
        print(f"    {action}: {count}")
    
    # Show sample with Tier 3 override
    tier3_override = [c for c in scam_concerns if any("Tier 3" in r for r in c.get('reasons', []))]
    if tier3_override:
        print(f"\n  Sample Tier-3 override (first 1):")
        item = tier3_override[0]
        print(f"    Index: {item.get('index')}")
        print(f"    Verdict: {item.get('tier3_verdict')}")
        print(f"    Reason: {item.get('tier3_reason')}")
        print(f"    Action: {item.get('recommended_action')}")
else:
    print("  (No scam concerns flagged in Week 24)")

print("\nOK: 3-tier fraud detection integration test complete")
print("\nOK: 3-tier fraud detection integration test complete")
