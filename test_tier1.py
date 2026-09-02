#!/usr/bin/env python3
"""Quick validation of Tier 1 scam detector with generic spam terms"""
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def main():
    try:
        print("=" * 70)
        print("TIER 1 VALIDATION")
        print("=" * 70)

        import app
        print("[OK] app.py imported")

        print("\n[TEST] Generic spam detection with URLs/links")
        print("-" * 70)

        test_cases = [
            ("Verify your account now by clicking here immediately: http://bit.ly/verify", 1, "Phishing + link"),
            ("Congratulations! You won a free prize. Claim your winnings now: http://bit.ly/claim", 1, "Lottery + shortlink"),
            ("Your package delivery on hold. Click to confirm: http://t.co/track", 1, "SMS spam + shortlink"),
            ("Make money fast with guaranteed returns. Send me a DM on WhatsApp to start now: http://earn.com", 1, "Financial spam + routing + link"),
            ("Send me a DM on WhatsApp for crypto gift card offer", 1, "BSF scam: crypto + routing"),
            ("Free giveaway! Send wire transfer now to claim. Limited time offer!", 1, "BSF scam: giveaway + payment + urgency"),
            ("On Tue, someone mentioned kernel maintainers in a newsletter thread.", 0, "Legit: eth substring should not trigger crypto"),
            ("Looking for PCS movers recommendations", 0, "Legit: PCS question"),
            ("VA benefits counselor available. Feel free to email.", 0, "Legit: veteran support"),
        ]

        results = []
        for text, expected, desc in test_cases:
            result = app.scam_heuristic_scan(text)
            score = result['heuristic_score']
            should_escalate, _ = app.classify_fraud_severity(result)
            escalate = 1 if should_escalate else 0
            match = escalate == expected
            results.append(match)

            status = "PASS" if match else "FAIL"
            print(f"[{status}] {desc}")
            print(f"      Score: {score:2d} | Got: {escalate} | Expected: {expected}")
            if result.get('matched_terms'):
                print(f"      Terms: {', '.join(result['matched_terms'][:2])}")

        accuracy = (sum(results) / len(results)) * 100
        print("\n" + "-" * 70)
        print(f"Accuracy: {sum(results)}/{len(results)} tests passed ({accuracy:.0f}%)")

        print("\n[CHECK] Generic spam term lists")
        print("-" * 70)
        for name in ['GENERIC_LOTTERY_SPAM', 'GENERIC_PHISHING', 'GENERIC_SMS_SPAM', 'GENERIC_FINANCIAL_SPAM']:
            has_it = hasattr(app, name)
            count = len(getattr(app, name)) if has_it else 0
            status = "OK" if has_it else "MISSING"
            print(f"[{status}] {name}: {count} terms")

        print("\n[CHECK] QA trusted-user override + NaN normalization + weekly scam path")
        print("-" * 70)
        qa_df = app.pd.DataFrame([
            {
                "content": "I just won the Lottery!!!! Email me at benjaminlouissaldana@gmail.com to let me know why you deserve $5,000 cash right now!",
                "commenter_name": "BSF Moderator",
                "commenter_email": "staff@bluestarfam.org",
                "community_name": "Test Forum",
            },
            {
                "content": "I have a business opportunity. Email me at mkang123@gmail.com I won the lottery",
                "commenter_name": "BSF Moderator",
                "commenter_email": float('nan'),
                "community_name": "Test Forum",
            },
        ])

        original_classifier = app._gemini_scam_classify_cached
        app._gemini_scam_classify_cached = lambda _text: '{"reasoning": "Mocked weekly test", "is_scam": true, "confidence": 0.96, "recommended_action": "block", "indicators": ["won", "email_address"]}'
        try:
            default_flags = app.detect_scam_concerns(qa_df, silent=True, include_trusted_users=False)
            qa_flags = app.detect_scam_concerns(qa_df, silent=True, include_trusted_users=True)
        finally:
            app._gemini_scam_classify_cached = original_classifier

        default_count = app.count_scam_tier1_candidates(qa_df, include_trusted_users=False)
        qa_count = app.count_scam_tier1_candidates(qa_df, include_trusted_users=True)
        normalized_email = app.normalize_identity_value(float('nan'))
        detected_email_ok = any(flag.get("detected_contact_email") == "mkang123@gmail.com" for flag in qa_flags)
        qa_checks_ok = (
            default_count == 0 and
            qa_count == 2 and
            len(default_flags) == 0 and
            len(qa_flags) == 2 and
            normalized_email == 'N/A' and
            detected_email_ok
        )
        print(
            f"[{'PASS' if qa_checks_ok else 'FAIL'}] tier1_default={default_count} | tier1_qa={qa_count} | "
            f"weekly_default={len(default_flags)} | weekly_qa={len(qa_flags)} | normalized_email={normalized_email} | detected_email_ok={detected_email_ok}"
        )
        results.append(qa_checks_ok)

        print("\n[CHECK] Gemini-unavailable fallback should still surface Tier-1 medium/high scams")
        print("-" * 70)
        fallback_df = app.pd.DataFrame([
            {
                "content": "Free giveaway! Send wire transfer now to claim. Limited time offer!",
                "commenter_name": "Community User",
                "commenter_email": "member@example.com",
                "community_name": "Test Forum",
            },
            {
                "content": "Looking for PCS movers recommendations",
                "commenter_name": "Community User",
                "commenter_email": "member2@example.com",
                "community_name": "Test Forum",
            },
        ])

        original_classifier = app._gemini_scam_classify_cached
        app._gemini_scam_classify_cached = lambda _text: "ERROR: GEMINI_UNAVAILABLE"
        try:
            fallback_flags = app.detect_scam_concerns(fallback_df, silent=True, include_trusted_users=True)
        finally:
            app._gemini_scam_classify_cached = original_classifier

        fallback_ok = len(fallback_flags) >= 1 and any(
            str(flag.get("llm_status", "")).startswith("ERROR:") for flag in fallback_flags
        )
        print(f"[{'PASS' if fallback_ok else 'FAIL'}] flags_when_gemini_down={len(fallback_flags)}")
        results.append(fallback_ok)

        print("\n[CHECK] Legacy SMS bait rule (recall boost with precision guard)")
        print("-" * 70)
        sms_scam = "You have WON a guaranteed £1000 cash prize. To claim call 09061743386 now"
        sms_legit = "Free community meetup this weekend. Call me at 555-123-4567 for details."
        sms_scam_h = app.scam_heuristic_scan(sms_scam)
        sms_legit_h = app.scam_heuristic_scan(sms_legit)
        sms_scam_escalate, sms_scam_sev = app.classify_fraud_severity(sms_scam_h)
        sms_legit_escalate, sms_legit_sev = app.classify_fraud_severity(sms_legit_h)
        sms_rule_ok = bool(sms_scam_escalate) and (not sms_legit_escalate)
        print(
            f"[{'PASS' if sms_rule_ok else 'FAIL'}] scam_score={sms_scam_h.get('heuristic_score')} sev={sms_scam_sev} | "
            f"legit_score={sms_legit_h.get('heuristic_score')} sev={sms_legit_sev}"
        )
        results.append(sms_rule_ok)

        print("\n[CHECK] Severity-path recall rules (shortcode smishing + account-security phishing)")
        print("-" * 70)
        shortcode_sms_text = "Last Chance! Claim ur £150 worth of discount vouchers today! Text SHOP to 85023 now!"
        acct_phish_text = "Dear bank member, unauthorized access detected after failed login attempts on your internet banking account."
        bank_awareness_text = "Our bank security team shared a member login checklist for awareness training."

        short_h = app.scam_heuristic_scan(shortcode_sms_text)
        short_escalate, short_sev = app.classify_fraud_severity(short_h)
        phish_h = app.scam_heuristic_scan(acct_phish_text)
        phish_escalate, phish_sev = app.classify_fraud_severity(phish_h)
        aware_h = app.scam_heuristic_scan(bank_awareness_text)
        aware_escalate, aware_sev = app.classify_fraud_severity(aware_h)

        severity_rules_ok = bool(short_escalate) and bool(phish_escalate) and (not aware_escalate)
        print(
            f"[{'PASS' if severity_rules_ok else 'FAIL'}] shortcode(score={short_h.get('heuristic_score')}, sev={short_sev}) | "
            f"account_phish(score={phish_h.get('heuristic_score')}, sev={phish_sev}) | "
            f"awareness(score={aware_h.get('heuristic_score')}, sev={aware_sev})"
        )
        results.append(severity_rules_ok)

        print("\n" + "=" * 70)
        if all(results):
            print("RESULT: Tier 1 policy validated successfully")
            return 0
        print(f"RESULT: Tier 1 policy mismatch detected - {sum(results)}/{len(results)} checks passed")
        return 1

    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
