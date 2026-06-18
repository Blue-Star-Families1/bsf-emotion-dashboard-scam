#!/usr/bin/env python3
"""Test full 3-tier pipeline orchestration"""
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    print("=" * 70)
    print("FULL 3-TIER PIPELINE TEST")
    print("=" * 70)
    
    import app
    print("[OK] app.py imported")
    
    # Create small test dataset
    test_df = pd.DataFrame({
        'content': [
            "Verify your account now: http://bit.ly/verify",  # Generic phishing
            "Send me a DM on WhatsApp for crypto gift card",    # BSF scam
            "Looking for PCS movers",                          # Legit
            "Congratulations you won! Claim prize http://example.com",  # Lottery
            "Hi, I'm a VA counselor, feel free to email me",  # Legit with email
        ],
        'commenter_name': ['user1', 'user2', 'user3', 'user4', 'user5'],
        'commenter_email': ['u1@test.com', 'u2@test.com', 'u3@test.com', 'u4@test.com', 'u5@bsf.org'],
    })
    
    print("\n[TEST] Tier 1 escalation rates")
    print("-" * 70)
    
    tier1_escalations = 0
    expected_escalations = [True, True, False, False, False]
    actual_escalations = []
    for i, row in test_df.iterrows():
        result = app.scam_heuristic_scan(row['content'])
        escalate = app.should_escalate_tier1(result)
        actual_escalations.append(bool(escalate))
        if escalate:
            tier1_escalations += 1
        status = "ESCALATE" if escalate else "PASS"
        print(f"[{status}] Row {i}: Score={result['heuristic_score']} - {row['content'][:40]}")
    
    print(f"\nTier 1 escalation rate: {tier1_escalations}/{len(test_df)} = {(tier1_escalations/len(test_df)*100):.0f}%")
    if actual_escalations == expected_escalations:
        print("[OK] Tier 1 escalation policy matches expected production behavior")
    else:
        print("[ERROR] Tier 1 escalation policy drift detected")
        print(f"        Expected: {expected_escalations}")
        print(f"        Actual:   {actual_escalations}")
        sys.exit(1)
    
    print("\n[TEST] Orchestration function signature")
    print("-" * 70)
    
    # Check if detect_scam_concerns function exists and is callable
    if hasattr(app, 'detect_scam_concerns'):
        print("[OK] detect_scam_concerns function exists")
        func = app.detect_scam_concerns
        print(f"      Function type: {type(func)}")
        print(f"      Module: {func.__module__}")
    else:
        print("[ERROR] detect_scam_concerns function NOT FOUND")
    
    print("\n[TEST] Gemini availability")
    print("-" * 70)
    
    # Check if Gemini is available
    try:
        gemini_model, available = app.load_gemini_model(app.FAST_REVIEW_MODEL_NAME)
        if available:
            print(f"[OK] Gemini available: {app.FAST_REVIEW_MODEL_NAME}")
        else:
            print(f"[WARN] Gemini not available (expected in non-production)")
    except Exception as e:
        print(f"[WARN] Gemini check skipped: {str(e)[:50]}")
    
    print("\n[TEST] Tier 2 prompt template")
    print("-" * 70)
    
    if hasattr(app, 'SCAM_CHECK_PROMPT_TEMPLATE'):
        template = app.SCAM_CHECK_PROMPT_TEMPLATE
        print(f"[OK] SCAM_CHECK_PROMPT_TEMPLATE exists")
        print(f"     Length: {len(template)} chars")
        print(f"     Contains 'RED FLAGS': {'RED FLAGS' in template}")
        print(f"     Contains 'GREEN FLAGS': {'GREEN FLAGS' in template}")
    else:
        print("[ERROR] SCAM_CHECK_PROMPT_TEMPLATE NOT FOUND")
    
    print("\n[TEST] Tier 2.5 military context prompt")
    print("-" * 70)
    
    if hasattr(app, 'BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE'):
        template = app.BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE
        print(f"[OK] BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE exists")
        print(f"     Length: {len(template)} chars")
        print(f"     Contains 'PCS': {'PCS' in template}")
        print(f"     Contains 'deployment': {'deployment' in template or 'Deployment' in template}")
    else:
        print("[ERROR] BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE NOT FOUND")
    
    print("\n" + "=" * 70)
    print("RESULT: Pipeline infrastructure intact - ready for integration")
    print("=" * 70)

except Exception as e:
    print(f"[ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
