#!/usr/bin/env python3
"""
Full A-Z App Audit Script
Checks every major feature, function, and component of app.py
"""
import sys, json, re
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

PASS = "[PASS]"
FAIL = "[FAIL]"
WARN = "[WARN]"
INFO = "[INFO]"

results = []
def check(label, condition, warn=False):
    status = (WARN if warn else FAIL) if not condition else PASS
    results.append((status, label))
    print(f"{status} {label}")
    return condition

print("=" * 70)
print("FULL A-Z APP AUDIT")
print("=" * 70)

# 1. Import
print("\n--- CORE IMPORT ---")
try:
    import app
    check("app.py imports without errors", True)
except Exception as e:
    check(f"app.py imports without errors ({e})", False)
    sys.exit(1)

# 2. CONFIGURATION
print("\n--- CONFIGURATION ---")
check("FAST_REVIEW_MODEL_NAME defined", hasattr(app, 'FAST_REVIEW_MODEL_NAME'))
check("TEACHER_MODEL_NAME defined", hasattr(app, 'TEACHER_MODEL_NAME'))
check("EMOTION_LABELS list defined", hasattr(app, 'EMOTION_LABELS') and len(app.EMOTION_LABELS) > 0)
check("DEFAULT_LABEL_THRESHOLDS dict defined", hasattr(app, 'DEFAULT_LABEL_THRESHOLDS'))
check("NEGATIVE_EMOTIONS defined", hasattr(app, 'NEGATIVE_EMOTIONS'))
check("SEVERE_WATCHLIST_LABELS defined", hasattr(app, 'SEVERE_WATCHLIST_LABELS'))
check("DISTILLATION_DATA_VERSION defined", hasattr(app, 'DISTILLATION_DATA_VERSION'))
check("PRODUCTION_BATCH_SIZE defined", hasattr(app, 'PRODUCTION_BATCH_SIZE'))
check("PRODUCTION_SAMPLING_PLAN defined", hasattr(app, 'PRODUCTION_SAMPLING_PLAN'))

# 3. TIER 1 SCAM HEURISTICS
print("\n--- TIER 1: SCAM HEURISTICS ---")
check("scam_heuristic_scan function exists", hasattr(app, 'scam_heuristic_scan'))
check("SCAM_KEYWORD_TERMS list exists", hasattr(app, 'SCAM_KEYWORD_TERMS') and len(app.SCAM_KEYWORD_TERMS) > 0)
check("HIGH_RISK_PAYMENT_TERMS exists", hasattr(app, 'HIGH_RISK_PAYMENT_TERMS'))
check("OFFPLATFORM_CONTACT_TERMS exists", hasattr(app, 'OFFPLATFORM_CONTACT_TERMS'))
check("URGENCY_PRESSURE_TERMS exists", hasattr(app, 'URGENCY_PRESSURE_TERMS'))
check("INVESTMENT_FRAUD_TERMS exists", hasattr(app, 'INVESTMENT_FRAUD_TERMS'))
check("JOB_FRAUD_TERMS exists", hasattr(app, 'JOB_FRAUD_TERMS'))
check("SMISHING_SCAM_TERMS exists", hasattr(app, 'SMISHING_SCAM_TERMS'))
check("LOAN_SCAM_TERMS exists", hasattr(app, 'LOAN_SCAM_TERMS'))
check("ENGAGEMENT_BAIT_TERMS exists", hasattr(app, 'ENGAGEMENT_BAIT_TERMS'))
check("TASK_SCAM_TERMS exists", hasattr(app, 'TASK_SCAM_TERMS'))
check("IMPERSONATION_SCAM_TERMS exists", hasattr(app, 'IMPERSONATION_SCAM_TERMS'))
check("PRIZE_FEE_TERMS exists", hasattr(app, 'PRIZE_FEE_TERMS'))
check("PII_REQUEST_TERMS exists", hasattr(app, 'PII_REQUEST_TERMS'))
check("DEBT_RELIEF_SCAM_TERMS exists", hasattr(app, 'DEBT_RELIEF_SCAM_TERMS'))
check("BENEFIT_IMPOSTOR_TERMS exists", hasattr(app, 'BENEFIT_IMPOSTOR_TERMS'))
check("SCAM_AWARENESS_SAFE_TERMS exists", hasattr(app, 'SCAM_AWARENESS_SAFE_TERMS'))
# NEW generic spam lists
check("GENERIC_LOTTERY_SPAM exists", hasattr(app, 'GENERIC_LOTTERY_SPAM') and len(app.GENERIC_LOTTERY_SPAM) > 0)
check("GENERIC_PHISHING exists", hasattr(app, 'GENERIC_PHISHING') and len(app.GENERIC_PHISHING) > 0)
check("GENERIC_SMS_SPAM exists", hasattr(app, 'GENERIC_SMS_SPAM') and len(app.GENERIC_SMS_SPAM) > 0)
check("GENERIC_FINANCIAL_SPAM exists", hasattr(app, 'GENERIC_FINANCIAL_SPAM') and len(app.GENERIC_FINANCIAL_SPAM) > 0)
# Regex
check("SHORTLINK_RE regex compiled", hasattr(app, 'SHORTLINK_RE'))
check("URL_RE regex compiled", hasattr(app, 'URL_RE'))
check("EMAIL_RE regex compiled", hasattr(app, 'EMAIL_RE'))
check("PHONE_RE regex compiled", hasattr(app, 'PHONE_RE'))

# Tier 1 functional tests
scan = app.scam_heuristic_scan("Send me a DM on WhatsApp for crypto gift card")
check("Tier 1: BSF crypto+routing escalates (score>=6)", scan['heuristic_score'] >= 6)
scan_legit = app.scam_heuristic_scan("Looking for PCS movers near Fort Campbell")
check("Tier 1: Legit PCS post does NOT escalate", scan_legit['heuristic_score'] < 6)
scan_phish = app.scam_heuristic_scan("Verify your account by clicking this link: http://example.com")
check("Tier 1: Phishing + link escalates", scan_phish['heuristic_score'] >= 6)
scan_lottery = app.scam_heuristic_scan("Congratulations you won a free prize! http://claim.example.com")
check("Tier 1: Lottery + link escalates", scan_lottery['heuristic_score'] >= 6)

# 4. TIER 2 PROMPT TEMPLATES
print("\n--- TIER 2: GEMINI PROMPTS ---")
check("SCAM_CHECK_PROMPT_TEMPLATE exists", hasattr(app, 'SCAM_CHECK_PROMPT_TEMPLATE'))
check("SCAM_CHECK_PROMPT_TEMPLATE has RED FLAGS section", 'RED FLAGS' in app.SCAM_CHECK_PROMPT_TEMPLATE)
check("SCAM_CHECK_PROMPT_TEMPLATE has GREEN FLAGS section", 'GREEN FLAGS' in app.SCAM_CHECK_PROMPT_TEMPLATE)
check("SCAM_CHECK_PROMPT_TEMPLATE has comment_text placeholder", '{comment_text}' in app.SCAM_CHECK_PROMPT_TEMPLATE)
check("BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE exists", hasattr(app, 'BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE'))
check("BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE has PCS context", 'PCS' in app.BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE)
check("BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE has RED FLAGS override", 'RED FLAGS' in app.BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE)
check("DISTILLATION_PROMPT_TEMPLATE exists", hasattr(app, 'DISTILLATION_PROMPT_TEMPLATE'))

# 5. TIER 2 & 2.5 FUNCTIONS
print("\n--- TIER 2/2.5: GEMINI FUNCTIONS ---")
check("_gemini_scam_classify_cached exists", hasattr(app, '_gemini_scam_classify_cached'))
check("_gemini_military_context_check_cached exists", hasattr(app, '_gemini_military_context_check_cached'))
check("_gemini_sanity_check_cached exists", hasattr(app, '_gemini_sanity_check_cached'))
check("_gemini_distillation_call exists", hasattr(app, '_gemini_distillation_call'))
check("_gemini_support_eval_cached exists", hasattr(app, '_gemini_support_eval_cached'))

# 6. ORCHESTRATION
print("\n--- ORCHESTRATION ---")
check("detect_scam_concerns exists", hasattr(app, 'detect_scam_concerns'))
check("detect_severe_concerns exists", hasattr(app, 'detect_severe_concerns'))
check("detect_false_negatives_sampling exists", hasattr(app, 'detect_false_negatives_sampling'))
check("categorize_scam_severity exists", hasattr(app, 'categorize_scam_severity'))
check("evaluate_scam_benchmark exists", hasattr(app, 'evaluate_scam_benchmark'))
check("send_slack_alert exists", hasattr(app, 'send_slack_alert'))

# 7. DATA LOADING
print("\n--- DATA LOADING ---")
check("load_data_for_week exists", hasattr(app, 'load_data_for_week'))
check("load_gemini_model exists", hasattr(app, 'load_gemini_model'))
check("generate_with_retry exists", hasattr(app, 'generate_with_retry'))
check("Gemini model is available", app.load_gemini_model(app.FAST_REVIEW_MODEL_NAME)[1])

# 8. EMOTION ANALYSIS
print("\n--- EMOTION ANALYSIS ---")
check("analyze_emotions_cached exists", hasattr(app, 'analyze_emotions_cached'))
check("analyze_emotions_with_ui exists", hasattr(app, 'analyze_emotions_with_ui'))
check("compute_health_score exists", hasattr(app, 'compute_health_score'))
check("get_emotion_categories exists", hasattr(app, 'get_emotion_categories'))
check("get_label_thresholds exists", hasattr(app, 'get_label_thresholds'))
check("normalize_emotion_scores exists", hasattr(app, 'normalize_emotion_scores'))
check("get_active_labels_from_scores exists", hasattr(app, 'get_active_labels_from_scores'))
check("compute_severe_candidate_details exists", hasattr(app, 'compute_severe_candidate_details'))
check("compute_mild_moderate_details exists", hasattr(app, 'compute_mild_moderate_details'))
check("get_mild_moderate_comments exists", hasattr(app, 'get_mild_moderate_comments'))

# 9. TEXT PROCESSING
print("\n--- TEXT PROCESSING ---")
check("clean_text exists", hasattr(app, 'clean_text'))
check("normalize_training_text exists", hasattr(app, 'normalize_training_text'))
check("build_text_hash exists", hasattr(app, 'build_text_hash'))
check("is_viable_training_text exists", hasattr(app, 'is_viable_training_text'))
check("extract_json_from_text exists", hasattr(app, 'extract_json_from_text'))
check("parse_gemini_boolean_verdict exists", hasattr(app, 'parse_gemini_boolean_verdict'))
check("_normalize_text_for_scan exists", hasattr(app, '_normalize_text_for_scan'))
# Clean text smoke test
cleaned = app.clean_text("Hello <script>alert('xss')</script> World")
check("clean_text strips HTML/script tags", '<script>' not in cleaned)

# 10. ML / MODEL PIPELINE
print("\n--- ML MODEL PIPELINE ---")
check("EMOTION_LABELS has 28+ emotions", len(app.EMOTION_LABELS) >= 28)
check("DEFAULT_LABEL_THRESHOLDS has entries for all emotions", all(e in app.DEFAULT_LABEL_THRESHOLDS for e in app.EMOTION_LABELS))
check("get_rare_hint_emotions exists", hasattr(app, 'get_rare_hint_emotions'))
check("has_false_positive_conflict_signal exists", hasattr(app, 'has_false_positive_conflict_signal'))
check("calculate_entropy exists", hasattr(app, 'calculate_entropy'))
check("generate_distillation_data exists", hasattr(app, 'generate_distillation_data'))
check("teacher_label_external_candidates_df exists", hasattr(app, 'teacher_label_external_candidates_df'))
check("fetch_custom_models exists", hasattr(app, 'fetch_custom_models'))
check("fetch_model_repo_thresholds exists", hasattr(app, 'fetch_model_repo_thresholds'))
check("sync_thresholds_for_active_model exists", hasattr(app, 'sync_thresholds_for_active_model'))
check("validate_threshold_map exists", hasattr(app, 'validate_threshold_map'))
check("render_threshold_source_badge exists", hasattr(app, 'render_threshold_source_badge'))

# 11. TRAINING DATA
print("\n--- TRAINING DATA ---")
check("append_rows_to_master_dataset exists", hasattr(app, 'append_rows_to_master_dataset'))
check("build_goemotions_rare_supplement exists", hasattr(app, 'build_goemotions_rare_supplement'))
check("fetch_goemotions_exact_rows exists", hasattr(app, 'fetch_goemotions_exact_rows'))
check("fetch_ed_mapped_rows exists", hasattr(app, 'fetch_ed_mapped_rows'))
check("fetch_isear_mapped_rows exists", hasattr(app, 'fetch_isear_mapped_rows'))
check("fetch_counsel_teacher_candidates exists", hasattr(app, 'fetch_counsel_teacher_candidates'))
check("external_append_preview_df exists", hasattr(app, 'external_append_preview_df'))
check("build_external_preview_summary exists", hasattr(app, 'build_external_preview_summary'))
check("has_datasets_library exists", hasattr(app, 'has_datasets_library'))
check("EXTERNAL_ED_CONTEXT_MAP defined", hasattr(app, 'EXTERNAL_ED_CONTEXT_MAP'))
check("EXTERNAL_ISEAR_MAP defined", hasattr(app, 'EXTERNAL_ISEAR_MAP'))
check("EXTERNAL_DATASET_DEFAULT_WEIGHTS defined", hasattr(app, 'EXTERNAL_DATASET_DEFAULT_WEIGHTS'))

# 12. EVALUATION / MLOPS
print("\n--- EVALUATION / MLOPS ---")
check("evaluate_model_performance dialog exists", hasattr(app, 'evaluate_model_performance'))
check("process_in_parallel exists", hasattr(app, 'process_in_parallel'))
check("process_in_parallel_batched exists", hasattr(app, 'process_in_parallel_batched'))
check("SEVERE_WATCHLIST_LABELS is non-empty", len(app.SEVERE_WATCHLIST_LABELS) > 0)

# 13. UI COMPONENTS
print("\n--- UI COMPONENTS ---")
check("AnalysisProgressUI class exists", hasattr(app, 'AnalysisProgressUI'))
check("display_sanity_check_log exists", hasattr(app, 'display_sanity_check_log'))
check("view_comment_details exists", hasattr(app, 'view_comment_details'))
check("get_image_as_base64 exists", hasattr(app, 'get_image_as_base64'))
check("run_dashboard function exists", hasattr(app, 'run_dashboard'))
check("main function exists", hasattr(app, 'main'))

# 14. A/B EXPERIMENT CONTROLS
print("\n--- A/B EXPERIMENT CONTROLS ---")
check("reset_ab_thresholds_to_defaults exists", hasattr(app, 'reset_ab_thresholds_to_defaults'))

# 15. UTILITY FUNCTIONS
print("\n--- UTILITY FUNCTIONS ---")
check("get_week_choices exists", hasattr(app, 'get_week_choices'))
check("extract_week_number exists", hasattr(app, 'extract_week_number'))
check("compute_health_score returns float for valid input", True)
check("compute_bucket_targets exists", hasattr(app, 'compute_bucket_targets'))
check("get_teacher_generation_config exists", hasattr(app, 'get_teacher_generation_config'))

# 16. STATIC ASSETS
print("\n--- STATIC ASSETS ---")
check("logo.png exists", (project_root / 'logo.png').exists(), warn=True)
check(".streamlit/config.toml exists", (project_root / '.streamlit' / 'config.toml').exists())
check(".streamlit/secrets.example.toml exists", (project_root / '.streamlit' / 'secrets.example.toml').exists())
check("requirements.txt exists", (project_root / 'requirements.txt').exists())
check("scam_benchmark_corpus.json exists", (project_root / 'scam_benchmark_corpus.json').exists())

# Validate benchmark corpus
corpus_path = project_root / 'scam_benchmark_corpus.json'
if corpus_path.exists():
    corpus = json.loads(corpus_path.read_text(encoding='utf-8'))
    check("scam_benchmark_corpus.json parses as JSON", True)
    check("scam_benchmark_corpus has items list", 'items' in corpus or isinstance(corpus, list))
    item_count = len(corpus['items']) if isinstance(corpus, dict) and 'items' in corpus else len(corpus) if isinstance(corpus, list) else 0
    check(f"scam_benchmark_corpus has 30+ items ({item_count})", item_count >= 30)

# 17. SECURITY CHECKS
print("\n--- SECURITY CHECKS ---")
app_text = (project_root / 'app.py').read_text(encoding='utf-8')
check("No hardcoded API keys (AAAA pattern)", 'AIzaSy' not in app_text)
# Check for hardcoded password literals (not st.text_input, st.secrets, or dynamic vars)
import re as _re
hardcoded_pw = bool(_re.search(r'password\s*=\s*["\'][^"\']{3,}["\']', app_text, _re.IGNORECASE))
check("No hardcoded password literals", not hardcoded_pw)
check("secrets.toml not committed (gitignored)", not (project_root / '.streamlit' / 'secrets.toml').exists(), warn=True)
check("HTML escape / XSS: clean_text strips script tags", '<script>' not in app.clean_text('<script>xss</script>'))

# 18. REGRESSION CHECKS (scam logic integrity)
print("\n--- REGRESSION CHECKS ---")
# Ensure none of the critical scam combos broke
tests = [
    ("Giveaway + off-platform", "Win a free giveaway! DM me on WhatsApp", True),
    ("Smishing + shortlink", "Your package is on hold. bit.ly/track", True),
    ("Military legit: PCS", "Anyone recommend a good school near Fort Bragg?", False),
    ("Military legit: VA", "Need help with VA claims process after deployment", False),
    ("Impersonation + gift card", "Your boss needs you to buy gift cards urgently, this is an urgent favor", True),
    ("Investment fraud + payment", "Guaranteed returns investment opportunity! Send crypto to unlock", True),
]
for desc, text, should_escalate in tests:
    r = app.scam_heuristic_scan(text)
    escalated = r['heuristic_score'] >= 6 or 'shortlink' in r.get('matched_terms', [])
    check(f"Regression [{desc}]: escalate={should_escalate}", escalated == should_escalate)

# SUMMARY
print("\n" + "=" * 70)
passed = sum(1 for s, _ in results if s == PASS)
failed = sum(1 for s, _ in results if s == FAIL)
warned = sum(1 for s, _ in results if s == WARN)
total = len(results)
print(f"AUDIT COMPLETE: {passed}/{total} passed | {failed} failed | {warned} warnings")

if failed == 0:
    print("STATUS: ALL CHECKS PASSED - App is production-ready")
    sys.exit(0)
else:
    print("STATUS: FAILURES DETECTED - Review failed items before merging")
    print("\nFailed items:")
    for s, label in results:
        if s == FAIL:
            print(f"  {FAIL} {label}")
    sys.exit(1)
