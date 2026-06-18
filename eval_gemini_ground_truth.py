"""
eval_gemini_ground_truth.py
────────────────────────────────────────────────────────────────
Run ALL comments for a given week directly through Gemini to
establish the TRUE count of BSF-concerning severely negative
comments, then compare against the 2-tier pipeline.

Usage:
  py -3.11 eval_gemini_ground_truth.py              # Week 24, 2025
  py -3.11 eval_gemini_ground_truth.py 23 2025
  py -3.11 eval_gemini_ground_truth.py 21 2025
"""
import sys, time, random, json, os, concurrent.futures, datetime
import pandas as pd

# ── python 3.11 ships tomllib in stdlib ──────────────────────
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("ERROR: need tomllib (built-in >=3.11) or 'pip install tomli'", flush=True)
        sys.exit(1)

# ── parse args ───────────────────────────────────────────────
week = int(sys.argv[1]) if len(sys.argv) > 1 else 24
year = int(sys.argv[2]) if len(sys.argv) > 2 else 2025
print(f"\n{'='*66}", flush=True)
print(f"  Gemini Ground-Truth Evaluation  --  Week {week}, {year}", flush=True)
print(f"{'='*66}\n", flush=True)

# ── load secrets from secrets.toml ───────────────────────────
_root = os.path.dirname(os.path.abspath(__file__))
SECRETS_PATH = os.path.join(_root, '.streamlit', 'secrets.toml')
try:
    with open(SECRETS_PATH, 'rb') as _f:
        _secrets = tomllib.load(_f)
except FileNotFoundError:
    print(f"ERROR: secrets file not found at {SECRETS_PATH}", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"ERROR reading secrets: {e}", flush=True)
    sys.exit(1)

GEMINI_API_KEY = (_secrets.get('gemini') or {}).get('api_key', '')
if not GEMINI_API_KEY:
    print("ERROR: gemini.api_key missing in secrets.toml", flush=True)
    sys.exit(1)
print("Gemini API key loaded OK.", flush=True)

# ── init Gemini SDK (modern preferred, legacy fallback) ───────
_gemini_client = None
_legacy_genai  = None
GEMINI_BACKEND = None
try:
    from google import genai as _modern_genai
    _gemini_client = _modern_genai.Client(api_key=GEMINI_API_KEY)
    GEMINI_BACKEND = 'modern'
    print("Gemini backend: google-genai (modern)", flush=True)
except Exception as _me:
    try:
        import google.generativeai as _legacy_genai
        _legacy_genai.configure(api_key=GEMINI_API_KEY)
        GEMINI_BACKEND = 'legacy'
        print("Gemini backend: google-generativeai (legacy)", flush=True)
    except Exception as _le:
        print(f"ERROR: could not load any Gemini SDK: {_me} / {_le}", flush=True)
        sys.exit(1)

FAST_MODEL   = 'gemini-3-flash-preview'   # must match app.py FAST_REVIEW_MODEL_NAME
MAX_CHARS    = 2500   # truncate very long posts to keep latency manageable
CALL_TIMEOUT = 25     # seconds per call before marking as error

SANITY_PROMPT = """
You are a Blue Star Families member-support reviewer. Decide whether a comment shows a **real member need, true concern, or intervention-worthy distress** that should be surfaced for staff review.

### Label "True" only when the comment includes at least one of:
- A concrete support/resource need (housing, finances, food, childcare, EFMP, medical, employment, PCS, relocation, safety, or family strain).
- Explicit personal distress, inability to cope, hopelessness, grief, or severe isolation.
- A direct or indirect request for help tied to a real hardship.
- A plausible urgent welfare concern in BSF context, even if understated.

### Label "False" when primarily:
- Informational, logistical, or a normal question without real hardship.
- Positive, appreciative, congratulatory, or celebratory.
- Casual conversation, networking, humor, or general military life discussion.
- Supportive of others but the author expresses no personal need.

### Important
- Do NOT label "True" for gratitude/caring tone alone.
- For BSF safety-review, when evidence is mixed but there is plausible hardship/resource need, lean toward True.

### Output format (REQUIRED)
REASONING: [Brief explanation]
VERDICT: [True or False]

Comment: "{comment_text}"
"""


def _call_gemini(text: str) -> str:
    snippet = text[:MAX_CHARS]
    prompt  = SANITY_PROMPT.format(comment_text=snippet)
    try:
        if GEMINI_BACKEND == 'modern':
            resp = _gemini_client.models.generate_content(model=FAST_MODEL, contents=prompt)
            return (resp.text or '').strip()
        else:
            mdl  = _legacy_genai.GenerativeModel(FAST_MODEL)
            resp = mdl.generate_content(prompt)
            return (resp.text or '').strip()
    except Exception as e:
        return f'ERROR: {e}'


def _call_with_timeout(text: str) -> str:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call_gemini, text)
        try:
            return fut.result(timeout=CALL_TIMEOUT)
        except concurrent.futures.TimeoutError:
            return 'ERROR: timeout (>25s)'


def _parse_verdict(raw: str):
    for line in (raw or '').splitlines():
        s = line.strip().upper()
        if s.startswith('VERDICT:'):
            val = s.split(':', 1)[1].strip()
            if val.startswith('TRUE'):  return True
            if val.startswith('FALSE'): return False
    return None


# ── warm-up: test one quick Gemini call ──────────────────────
print("Testing Gemini connectivity ...", flush=True)
_test = _call_with_timeout("Hello, are you available?")
if _test.startswith('ERROR'):
    print(f"WARNING: Gemini test returned: {_test[:120]}", flush=True)
    print("Continuing anyway -- errors counted per comment.", flush=True)
else:
    print("Gemini OK.\n", flush=True)

# ── Load Snowflake data DIRECTLY (bypasses app.py, hard timeout)
def _get_week_dates(year: int, week: int):
    jan4       = datetime.date(year, 1, 4)
    week_start = jan4 + datetime.timedelta(weeks=week - 1, days=-(jan4.weekday()))
    week_end   = week_start + datetime.timedelta(days=6)
    return str(week_start), str(week_end)


def _load_data_direct(week: int, year: int) -> pd.DataFrame:
    from cryptography.hazmat.primitives import serialization
    import snowflake.connector

    sf = _secrets.get('snowflake', {})
    passphrase = sf.get('private_key_passphrase') or None
    p_key_pem  = sf.get('private_key', '')

    private_key = serialization.load_pem_private_key(
        p_key_pem.encode(),
        password=passphrase.encode() if passphrase else None,
    )
    pkb = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    start_date, end_date = _get_week_dates(year, week)
    print(f"  Week dates: {start_date} -> {end_date}", flush=True)

    conn = snowflake.connector.connect(
        account=sf['account'],
        user=sf['user'],
        warehouse=sf.get('warehouse', ''),
        database=sf.get('database', ''),
        schema=sf.get('schema', ''),
        private_key=pkb,
        login_timeout=90,
        network_timeout=90,
        client_session_keep_alive=False,
    )
    try:
        cur = conn.cursor()
        query = """
            WITH Weekly_Posts AS (
                SELECT
                    'Original Post' AS activity_type,
                    Forum_Created_At AS activity_date,
                    CONCAT(IFNULL(Forum_Title, ''), ' - ', IFNULL(Forum_Content, '')) AS content,
                    community_name,
                    commenter_name,
                    commenter_email
                FROM BSF01_DEV.BSF01_NEIGHBORHOOD.VW_FORUM_DISCUSSIONS_MERGE
                WHERE TO_DATE(Forum_Created_At) BETWEEN %(start_date)s AND %(end_date)s
                  AND Forum_Content IS NOT NULL
                QUALIFY ROW_NUMBER() OVER (PARTITION BY Forum_Title, Forum_Content ORDER BY Forum_Created_At) = 1
            ),
            Weekly_Comments AS (
                SELECT
                    'Comment' AS activity_type,
                    Discussion_Created_At AS activity_date,
                    Discussion_Content AS content,
                    community_name,
                    commenter_name,
                    commenter_email
                FROM BSF01_DEV.BSF01_NEIGHBORHOOD.VW_FORUM_DISCUSSIONS_MERGE
                WHERE TO_DATE(Discussion_Created_At) BETWEEN %(start_date)s AND %(end_date)s
                  AND discussion_id IS NOT NULL
                  AND Discussion_Content IS NOT NULL
                  AND TRIM(Discussion_Content) != ''
                QUALIFY ROW_NUMBER() OVER (PARTITION BY discussion_id ORDER BY Discussion_Created_AT DESC) = 1
            )
            SELECT * FROM Weekly_Posts
            UNION ALL
            SELECT * FROM Weekly_Comments
        """
        cur.execute(query, {'start_date': start_date, 'end_date': end_date})
        rows    = cur.fetchall()
        columns = [desc[0].lower() for desc in cur.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


def _load_with_sf_timeout(week: int, year: int, hard_timeout: int = 300) -> pd.DataFrame:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_load_data_direct, week, year)
        try:
            return fut.result(timeout=hard_timeout)
        except concurrent.futures.TimeoutError:
            print(f"ERROR: Snowflake timed out after {hard_timeout}s", flush=True)
            return pd.DataFrame()
        except Exception as e:
            print(f"ERROR: Snowflake load failed: {e}", flush=True)
            return pd.DataFrame()


print("Loading Snowflake data (direct connector, 90s login + 300s hard timeout) ...", flush=True)
df = _load_with_sf_timeout(week, year, hard_timeout=300)
n  = len(df)
print(f"Loaded {n} interactions.", flush=True)
if n == 0:
    print("ERROR: No data returned. Exiting.", flush=True)
    sys.exit(1)

# ── import app for pipeline + staff filter ───────────────────
import io as _io
print("Importing app module (for staff filter + pipeline) ...", flush=True)
_old_stderr = sys.stderr
sys.stderr = _io.StringIO()
try:
    import app  # noqa: E402
finally:
    sys.stderr = _old_stderr

# ── STEP 1: Run Gemini on ALL comments ───────────────────────
print(f"\n[STEP 1] Gemini full-pass on all {n} comments ...", flush=True)
print(f"         Model={FAST_MODEL}  Timeout={CALL_TIMEOUT}s  MaxChars={MAX_CHARS}\n", flush=True)

gemini_true  = []
gemini_false = []
gemini_error = []

for seq, (idx, row) in enumerate(df.iterrows(), start=1):
    commenter = str(row.get('commenter_name', 'N/A'))
    email     = str(row.get('commenter_email', ''))
    text_body = str(row.get('content', ''))

    # Skip staff/admin
    if app.is_staff_or_admin_comment(commenter, email):
        gemini_false.append({
            'index': seq, 'commenter_name': commenter,
            'verdict': False, 'reason': 'staff_skip'
        })
        if seq % 20 == 0 or seq == n:
            print(f"  {seq}/{n}  true={len(gemini_true)}", flush=True)
        continue

    raw     = _call_with_timeout(text_body)
    verdict = _parse_verdict(raw)

    entry = {
        'index': seq, 'commenter_name': commenter, 'email': email,
        'text': text_body[:140], 'raw': raw[:220]
    }
    if verdict is True:
        gemini_true.append(entry)
        print(f"  [{seq:3d}] TRUE   {commenter[:28]:<28} | {text_body[:65]}", flush=True)
    elif verdict is False:
        gemini_false.append(entry)
    else:
        gemini_error.append(entry)
        print(f"  [{seq:3d}] ERROR  {raw[:70]}", flush=True)

    if seq % 20 == 0 or seq == n:
        print(f"  {seq}/{n}  true={len(gemini_true)}", flush=True)

    time.sleep(0.06 + random.uniform(0, 0.04))

print(f"\n[STEP 1 RESULT]", flush=True)
print(f"  Gemini TRUE  (BSF concerning) : {len(gemini_true)}", flush=True)
print(f"  Gemini FALSE (not concerning) : {len(gemini_false)}", flush=True)
print(f"  Parse errors / timeouts       : {len(gemini_error)}", flush=True)

# ── STEP 2: Run 2-tier pipeline (timeout-safe) ──────────────
print(f"\n[STEP 2] Running 2-tier RoBERTa + Gemini pipeline (timeout-safe) ...", flush=True)

pipeline_tier1 = []
pipeline_verified = []

# Tier-0: RoBERTa inference directly (bypass app.detect_severe_concerns timeout path)
roberta_model_id = 'SamLowe/roberta-base-go_emotions'
classifier, huggingface_available = app.load_emotion_classifier(roberta_model_id)
if not huggingface_available:
    print("  ERROR: RoBERTa model not available, skipping STEP 2", flush=True)
else:
    texts = df['content'].astype(str).str.slice(0, MAX_CHARS).tolist()
    rows = []
    batch_size = 32
    n_batches = (len(texts) + batch_size - 1) // batch_size

    print(f"  RoBERTa input truncated to {MAX_CHARS} chars/comment", flush=True)

    for bi in range(0, len(texts), batch_size):
        batch = texts[bi:bi + batch_size]
        rows.extend(app._run_inference_with_chunking(batch, classifier))
        print(f"  RoBERTa batch {bi // batch_size + 1}/{n_batches} done", flush=True)

    df_em = pd.DataFrame(rows).fillna(0)
    print(f"  RoBERTa inference done: {len(df_em)} rows", flush=True)

    # Tier-1: identify severe candidates (same heuristic as app)
    for i, em_row in df_em.iterrows():
        original_row = df.iloc[i]
        if app.is_staff_or_admin_comment(
            original_row.get('commenter_name', ''),
            original_row.get('commenter_email', '')
        ):
            continue

        comment_text = original_row.get('content', '')
        details = app.compute_severe_candidate_details(em_row, comment_text)
        if details['candidate']:
            pipeline_tier1.append({
                'index': i + 1,
                'commenter_name': original_row.get('commenter_name', 'N/A'),
                'comment': comment_text,
                'emotion': details.get('dominant_emotion', ''),
                'score': details.get('dominant_score', 0.0),
                'reason_flags': details.get('reason_flags', []),
            })

    print(f"  Tier-1 candidates: {len(pipeline_tier1)}", flush=True)

    # Tier-2: timeout-safe Gemini sanity check
    for item in pipeline_tier1:
        raw = _call_with_timeout(item['comment'])
        verdict = _parse_verdict(raw)
        status = 'VERIFIED' if verdict is True else ('DROPPED' if verdict is False else 'ERROR')
        print(
            f"  Tier-2 [{item['index']:3d}] {item['commenter_name'][:22]:<22} -> {status}: {raw[:55]}",
            flush=True
        )
        if verdict is True:
            pipeline_verified.append(item)

    print(f"  Verified (Tier-2): {len(pipeline_verified)}", flush=True)

# Both pipeline and our eval use 1-based sequential position
tier1_seq_set    = {item.get('index', -1) for item in pipeline_tier1}
verified_seq_set = {item.get('index', -1) for item in pipeline_verified}
gemini_true_seqs = {item['index'] for item in gemini_true}

# ── STEP 3: Compute metrics ──────────────────────────────────
true_positives  = verified_seq_set & gemini_true_seqs
false_positives = verified_seq_set - gemini_true_seqs
false_negatives = gemini_true_seqs - verified_seq_set
tier1_missed    = gemini_true_seqs - tier1_seq_set

precision = len(true_positives) / max(len(verified_seq_set), 1)
recall    = len(true_positives) / max(len(gemini_true_seqs), 1)
f1        = 2 * precision * recall / max(precision + recall, 1e-9)

print(f"\n{'='*66}", flush=True)
print(f"  PIPELINE EVALUATION  --  Week {week}, {year}", flush=True)
print(f"{'='*66}", flush=True)
print(f"  Gemini ground-truth BSF severe    : {len(gemini_true)}", flush=True)
print(f"  Pipeline Tier-1 candidates        : {len(pipeline_tier1)}", flush=True)
print(f"  Pipeline Gemini-verified (Tier-2) : {len(pipeline_verified)}", flush=True)
print(f"", flush=True)
print(f"  True  positives  (correct flags)  : {len(true_positives)}", flush=True)
print(f"  False positives  (over-flags)     : {len(false_positives)}", flush=True)
print(f"  False negatives  (missed)         : {len(false_negatives)}", flush=True)
print(f"    -- missed at Tier-1 (pre-Gemini): {len(tier1_missed)}", flush=True)
print(f"", flush=True)
print(f"  Precision  : {precision:.2%}", flush=True)
print(f"  Recall     : {recall:.2%}", flush=True)
print(f"  F1 score   : {f1:.2%}", flush=True)
print(f"{'='*66}\n", flush=True)

# ── Detail: false negatives (missed by pipeline) ─────────────
if false_negatives:
    print("FALSE NEGATIVES -- Gemini says BSF-concerning, pipeline missed:\n", flush=True)
    for item in sorted(gemini_true, key=lambda x: x['index']):
        if item['index'] in false_negatives:
            loc = "Tier-1 missed" if item['index'] in tier1_missed else "Tier-1 OK, Gemini2 dropped"
            print(f"  [{item['index']}] {item['commenter_name']} ({loc})", flush=True)
            print(f"       text : {item['text']}", flush=True)
            print(f"       gemini: {item['raw'][:120]}", flush=True)
            print(flush=True)

# ── Detail: false positives (pipeline over-flagged) ──────────
if false_positives:
    print("FALSE POSITIVES -- pipeline flagged, Gemini says NOT concerning:\n", flush=True)
    for item in pipeline_verified:
        if item.get('index', -1) in false_positives:
            print(f"  [{item.get('index')}] {item.get('commenter_name','N/A')} | emotion: {item.get('emotion','')}", flush=True)
            print(f"       flags: {item.get('reason_flags', [])}", flush=True)
            print(f"       {str(item.get('comment',''))[:120]}", flush=True)
            print(flush=True)

# ── Save JSON results ────────────────────────────────────────
results = {
    'week': week, 'year': year, 'total_comments': n,
    'gemini_true_count': len(gemini_true),
    'pipeline_tier1': len(pipeline_tier1),
    'pipeline_verified': len(pipeline_verified),
    'precision': round(precision, 4),
    'recall':    round(recall,    4),
    'f1':        round(f1,        4),
    'true_positives':  sorted(true_positives),
    'false_positives': sorted(false_positives),
    'false_negatives': sorted(false_negatives),
    'tier1_missed':    sorted(tier1_missed),
    'gemini_true_detail': gemini_true,
}
out_path = os.path.join(_root, f'eval_week{week}_{year}.json')
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved -> {out_path}", flush=True)
