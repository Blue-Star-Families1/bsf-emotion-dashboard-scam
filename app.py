# Fix OpenMP conflict error
import os
import sys
from pathlib import Path
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# Prevent optional torchvision import path from breaking text-only transformers pipeline on some cloud images.
os.environ.setdefault('TRANSFORMERS_NO_TORCHVISION', '1')

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
import math
import warnings
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from sqlalchemy import text, create_engine
import html
import base64
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time 
import random
import re
import json
import hashlib
import io
import concurrent.futures
from sklearn.metrics import f1_score, cohen_kappa_score, confusion_matrix, precision_recall_fscore_support
from huggingface_hub import HfApi, InferenceClient, hf_hub_download

modern_genai = None
modern_genai_types = None
legacy_genai = None
GEMINI_BACKEND = "unavailable"
_GEMINI_IMPORT_ATTEMPTED = False


def ensure_gemini_backend_loaded():
    """Lazy-load Gemini SDKs so optional dependency issues don't break app import/tests."""
    global modern_genai, modern_genai_types, legacy_genai, GEMINI_BACKEND, _GEMINI_IMPORT_ATTEMPTED

    if _GEMINI_IMPORT_ATTEMPTED:
        return GEMINI_BACKEND

    _GEMINI_IMPORT_ATTEMPTED = True

    try:
        from google import genai as _modern_genai
        from google.genai import types as _modern_genai_types
        modern_genai = _modern_genai
        modern_genai_types = _modern_genai_types
        GEMINI_BACKEND = "modern"
        return GEMINI_BACKEND
    except BaseException as modern_error:
        modern_genai = None
        modern_genai_types = None
        print(f"Modern Gemini SDK unavailable: {modern_error}", file=sys.stderr)

    try:
        import google.generativeai as _legacy_genai
        legacy_genai = _legacy_genai
        GEMINI_BACKEND = "legacy"
    except BaseException as legacy_error:
        legacy_genai = None
        GEMINI_BACKEND = "unavailable"
        print(f"Legacy Gemini SDK unavailable: {legacy_error}", file=sys.stderr)

    return GEMINI_BACKEND


class GeminiResponseAdapter:
    def __init__(self, text=""):
        self.text = text or ""


class GeminiModelAdapter:
    def __init__(self, backend, model_name, api_key):
        self.backend = backend
        self.model_name = model_name
        if backend == "modern":
            self.client = modern_genai.Client(api_key=api_key)
            self.model = None
        elif backend == "legacy":
            legacy_genai.configure(api_key=api_key)
            self.client = None
            self.model = legacy_genai.GenerativeModel(model_name)
        else:
            raise RuntimeError("Gemini backend is unavailable")

    @staticmethod
    def _safe_extract_text(response):
        text = getattr(response, "text", None)
        if text:
            return text

        candidates = getattr(response, "candidates", None) or []
        parts = []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            if content is None:
                continue
            content_parts = getattr(content, "parts", None) or []
            for part in content_parts:
                part_text = getattr(part, "text", None)
                if part_text:
                    parts.append(part_text)
        return "\n".join(parts)

    def generate_content(self, prompt, generation_config=None):
        generation_config = generation_config or {}

        if self.backend == "modern":
            config_obj = None
            if generation_config:
                config_obj = modern_genai_types.GenerateContentConfig(
                    response_mime_type=generation_config.get("response_mime_type"),
                    temperature=generation_config.get("temperature"),
                )
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config_obj,
            )
            return GeminiResponseAdapter(self._safe_extract_text(response))

        if self.backend == "legacy":
            config_obj = legacy_genai.GenerationConfig(**generation_config) if generation_config else None
            response = self.model.generate_content(prompt, generation_config=config_obj)
            return GeminiResponseAdapter(self._safe_extract_text(response))

        raise RuntimeError("Gemini backend is unavailable")


def build_generation_config(response_mime_type=None, temperature=None):
    config = {}
    if response_mime_type is not None:
        config["response_mime_type"] = response_mime_type
    if temperature is not None:
        config["temperature"] = temperature
    return config

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def detect_acceleration_device():
    """Safely detect inference acceleration without importing torch at app startup."""
    try:
        import torch

        if torch.cuda.is_available():
            return 0, "GPU"

        has_mps = (
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if has_mps:
            return "mps", "MPS (Apple Silicon)"
    except Exception:
        pass

    return -1, "CPU"


EMOTION_LABELS = [
    'sadness','anger','disgust','fear','annoyance','disappointment','disapproval','grief',
    'embarrassment','confusion','nervousness','remorse','neutral','joy','love','optimism',
    'admiration','gratitude','approval','amusement','excitement','pride','relief','caring',
    'desire','curiosity','surprise','realization'
]

DEFAULT_LABEL_THRESHOLDS = {
    'admiration': 0.25, 
    'amusement': 0.45, 
    'anger': 0.15, 
    'annoyance': 0.10, 
    'approval': 0.30,
    'caring': 0.40, 
    'confusion': 0.55, 
    'curiosity': 0.25, 
    'desire': 0.25, 
    'disappointment': 0.40,
    'disapproval': 0.30, 
    'disgust': 0.20, 
    'embarrassment': 0.10, 
    'excitement': 0.35, 
    'fear': 0.40,
    'gratitude': 0.45,
    # grief/relief/pride: model outputs very low scores for these (F1≈0.33 even at
    # optimal threshold). Set exactly to model-card optimal so they register at all.
    'grief': 0.05,
    'joy': 0.40,
    'love': 0.25,
    'nervousness': 0.25,
    'optimism': 0.20,
    'pride': 0.10,
    'realization': 0.15,
    'relief': 0.05,
    'remorse': 0.10,
    'sadness': 0.40, 
    'surprise': 0.15, 
    'neutral': 0.25
}

NEGATIVE_EMOTIONS = EMOTION_LABELS[:12]
SERIOUS_NEGATIVE_EMOTIONS = ['sadness', 'anger', 'fear', 'grief', 'remorse', 'nervousness', 'embarrassment', 'disappointment', 'disapproval']
FOCUS_RARE_EMOTIONS = ['grief', 'relief', 'pride', 'fear', 'remorse', 'nervousness', 'realization']
FALSE_POSITIVE_EMOTIONS = ['gratitude', 'caring', 'approval', 'excitement', 'admiration', 'optimism']

PRODUCTION_SAMPLING_PLAN = [
    ('rare_focus', 0.30),
    ('uncertain', 0.25),
    ('false_positive_suspect', 0.20),
    ('neutral_baseline', 0.15),
    ('random_baseline', 0.10),
]

PRODUCTION_BATCH_SIZE = 200
DISTILLATION_DATA_VERSION = 'v3_production_2000_sampler'
TEACHER_MODEL_NAME = 'gemini-3.1-pro-preview'
FAST_REVIEW_MODEL_NAME = 'gemini-3-flash-preview'
THRESHOLD_FILE_PATH = 'recommended_thresholds.json'
SCAM_BENCHMARK_FILE = Path(__file__).with_name("scam_benchmark_corpus.json")
SCAM_TIER1_SCORE_THRESHOLD = 10
SCAM_TIER1_SHORTLINK_TOKEN = "shortlink"
SHOW_FALSE_NEGATIVE_AUDIT_SECTIONS = False

ANALYSIS_STEP_LABELS = [
    'Emotion Analysis (RoBERTa + Sliding Window)',
    'AI Sanity Check (Gemini Chain-of-Thought)',
    'Scam Check (Tier 1: Deterministic Heuristics)',
    'Scam Check (Tier 2: LLM Verification)'
]

WEEKLY_DATA_COLUMNS = [
    'activity_type',
    'activity_date',
    'content',
    'community_name',
    'commenter_name',
    'commenter_email',
]

RARE_EMOTION_HINTS = {
    'grief': [r'\bpassed away\b', r'\bfuneral\b', r'\bmourning\b', r'\bwidow(?:ed)?\b', r'\bloss of\b', r'\bgrieving\b'],
    'relief': [r'\bso relieved\b', r'\bwhat a relief\b', r'\bfinally approved\b', r'\bfinally resolved\b', r'\bthank god\b'],
    'pride': [r'\bproud of\b', r'\bgraduated\b', r'\bpromotion\b', r'\bachievement\b', r'\baccomplished\b'],
    'fear': [r'\bafraid\b', r'\bterrified\b', r'\bscared\b', r'\bunsafe\b', r'\bin danger\b'],
    'remorse': [r'\bi regret\b', r"\bshouldn't have\b", r'\bwish i had not\b', r'\bmy fault\b'],
    'nervousness': [r'\bnervous\b', r'\banxious\b', r'\bpanic\b', r'\bworried\b', r'\bstressed\b'],
    'realization': [r'\bi realized\b', r'\bit dawned on me\b', r'\bnow i know\b', r'\bturns out\b']
}

FALSE_POSITIVE_CONFLICT_PATTERNS = [
    r'\bneed help\b', r'\bstruggling\b', r'\bworried\b', r'\bstressed\b', r'\boverwhelmed\b',
    r'\bcan\'t afford\b', r'\bunsafe\b', r'\bissue\b', r'\bproblem\b', r'\bfrustrat',
    r'\bconfused\b', r'\blooking for advice\b', r'\bany recommendations\b', r'\bcan anyone help\b',
    r'\bi need\b', r'\bhelp me\b', r'\bnot sure\b', r'\bdifficult\b', r'\bhard time\b'
]

SEVERE_REVIEW_THRESHOLDS = {
    'sadness': 0.30,
    'anger': 0.20,
    'fear': 0.20,
    # grief: model rarely outputs scores above 0.10 (optimal F1 threshold is 0.05
    # per model card). Set triage to 0.05 so any meaningful grief signal reaches Gemini.
    'grief': 0.05,
    'remorse': 0.10,
    'nervousness': 0.15,
    'embarrassment': 0.10,
    'disappointment': 0.25,
    'disapproval': 0.25,
    'relief': 0.05,
    'confusion': 0.45,
}

SEVERE_SUM_EMOTIONS = ['sadness', 'fear', 'grief', 'remorse', 'nervousness', 'embarrassment', 'disappointment']
SUPPORT_NEED_CONTEXT_EMOTIONS = ['sadness', 'fear', 'nervousness', 'disappointment', 'confusion']
SEVERE_WATCHLIST_LABELS = {'Grief', 'Fear', 'Sadness', 'Remorse', 'Nervousness', 'Embarrassment', 'Disappointment', 'Relief'}
SEVERE_SUM_HARD_THRESHOLD = 0.55
SEVERE_SPIKE_HARD_THRESHOLD = 0.40

SUPPORT_NEED_PATTERNS = [
    # Explicit direct asks
    r'\bneed help\b', r'\bcan anyone help\b', r'\bcan you help\b', r'\bplease help\b',
    r'\blooking for advice\b', r'\bneed support\b', r'\bneed resources\b', 
    r'\bnot sure what to do\b', r'\bnot sure where to turn\b', r'\bany suggestions\b',
    r'\bany recommendations\b', r'\brecommend\b', r'\bdoes anyone know\b',
    # Implicit hardship signals (indirect asks)
    r'\bstruggling\b', r'\boverwhelmed\b', r'\bstruggle\b', r'\bdifficult\b', r'\bhard time\b',
    r'\bworried\b', r'\banxious\b', r'\bstressed\b', r'\bstress\b', r'\bdoing okay\b',
    r'\bhold up\b', r'\bkeep going\b', r'\bsurviv\b', r'\bfalling apart\b',
    # Safety and vulnerability keywords
    r'\bunsafe\b', r'\bsafe\b', r'\bdanger\b', r'\bhomeless\b', r'\bdisabled\b',
    r'\bautistic\b', r'\bneurodiverse\b', r'\bifmp\b', r'\befmp\b', r'\bveteran benefit\b',
    # Financial hardship
    r"\bcan't afford\b", r'\bcannot afford\b', r'\bbehind on bills\b', r'\bbroke\b',
    r'\bfinancial hardship\b', r'\bpay\b', r'\bloan\b', r'\bdeb\b',
    r'\bhousing\b', r'\brent\b', r'\beviction\b', r'\bfood insecurity\b', r'\bfood bank\b',
    r'\bchildcare\b', r'\bdaycare\b', r'\bafford child\b',
    # Employment
    r'\bjob\b', r'\bunemploy', r'\bwork\b', r'\bcareer\b', r'\bopportunity\b',
    # Health/mental health
    r'\bmental health\b', r'\btherapy\b', r'\bdepressed\b', r'\bdepression\b',
    r'\bburned out\b', r'\bburnt out\b', r'\bpanic\b', r'\banxiety attack\b',
    r'\bgrieving\b', r'\bgrief\b', r'\bgone\b',
    # Loss/bereavement
    r'\bpassed away\b', r'\bfuneral\b', r'\blost my\b', r'\bwidow(?:ed)?\b', r'\bloss\b',
    # Military family stress
    r'\bdeployment\b', r'\bpcs\b', r'\brelocation\b', r'\bfamily strain\b', r'\bmarital strain\b',
    r'\bseparation\b', r'\bdisconnected\b', r'\biso(?:late)?d\b',
    # Financial/employment hardship (more specific)
    r'\bjob loss\b', r'\blayoff\b', r'\blaid off\b', r'\bbeen tough\b', r"\bit'?s been tough\b",
    r'\bcaregiver\b', r'\bcaretaker\b', r'\btough time\b', r'\bhard times\b',
    r'\bmaking ends meet\b', r'\bnot making ends\b', r'\bstretched thin\b'
]


# ==========================================
# --- PROMPTS (Upgraded with Chain-of-Thought) ---
# ==========================================

SANITY_CHECK_PROMPT_TEMPLATE = """
You are a Blue Star Families member-support reviewer. Your job is to decide whether a comment shows a **real member need, true concern, or intervention-worthy distress** that should be surfaced for staff review.

### Label "True" only when the comment includes at least one of the following:
- A concrete support or resource need (for example housing, finances, food, childcare, EFMP, medical, employment, PCS, relocation, safety, or family strain).
- Explicit personal distress, inability to cope, acute anxiety, panic, hopelessness, grief, or severe isolation.
- A direct or indirect request for help, advice, resources, or someone to listen that is tied to a real hardship.
- A plausible urgent welfare concern in BSF context, even if the emotion is understated.

### Label "False" when the comment is primarily:
- Informational, logistical, or a normal question without real hardship.
- Positive, appreciative, congratulatory, or celebratory.
- Casual conversation, networking, humor, recommendations, or general military life discussion without personal distress.
- Supportive language toward others without the author expressing a personal need.

### Important instructions
- Do **not** label "True" just because the text contains gratitude, caring, politeness, or emotional tone.
- Do **not** label "True" for generic updates, routine questions, or informational requests unless the member also expresses a real need or distress.
- For BSF safety-review context, when evidence is mixed but there is plausible hardship/resource need, lean toward **True** (staff can down-rank later).
- Prefer safety-aware precision: avoid obvious false positives, but do not miss likely member needs.

### Output Format
You must provide a brief reasoning followed by your final verdict.
Format your response exactly as follows:

REASONING: [Brief explanation of whether the comment shows a real member need, true concern, or intervention-worthy distress]
VERDICT: [True or False]

Comment: "{comment_text}"
"""

SCAM_CHECK_PROMPT_TEMPLATE = """
You are a trust & safety reviewer for Blue Star Families. Classify whether a user comment is likely a malicious scam, fraud, or predatory solicitation.

RED FLAGS (Likely Scams):
- Bait: "giving away", "free MacBook", "you've won"
- Predatory off-platform routing: "DM me for the money", WhatsApp/Telegram numbers, "kindly email me"
- Financial fraud: gift cards, wire, crypto, Cash App/Venmo/Zelle, "cash flip", "guaranteed returns"
- Phishing: "verify your account", "reset password", urgent links.
- Romance/Charity: "sugar daddy", fake orphanages.
- Smishing / phishing texts: package delivery issues, toll/traffic notices, tax refund claims, expiring reward points, redelivery fees, or unexpected loan approvals that push a link or ask for private info.
- Job/task scams: fake recruiters, "work from home" or "remote position" offers with vague duties, requests to reply YES/INTERESTED, WhatsApp/Telegram follow-up, task "optimization", or deposits/crypto needed to unlock earnings.
- Impersonation scams: messages pretending to be a boss, bank, government office, VA helper, or fraud department asking for gift cards, PINs, a "safe account" transfer, gold purchases, or secrecy.
- Prize / grant scams: unexpected winnings, sweepstakes, grants, or benefits that require taxes, shipping, handling, filing, or processing fees.

GREEN FLAGS & SAFE HARBORS (NOT Scams - DO NOT FLAG):
- Genuine Community Networking: Users asking for information, following up on a post, or asking how to get involved.
- Older User Behavior: Users signing their posts like a formal letter with their real name, phone number, and email address. This is normal, not a scam.
- Legitimate PCS/Military Services: Users offering or asking about moving services, driving, housing, or local businesses, especially if they provide detailed, coherent military context.
- Networking & Business: Users sharing their own business, offering legitimate services, or asking for local recommendations. 
- Peer Support: Do NOT flag users simply for sharing their email or phone number if they are offering peer support, mentorship, or answering a question.
- Legitimate jobs and services: concrete duties, normal application flow, official websites, no upfront payment, no gift cards, no crypto deposits, and no request to move money.
- Official BSF Admin posts.

CRITICAL INSTRUCTION:
If the user is offering a service, asking for advice, or sharing contact info for networking/support, IT IS NOT A SCAM. Scams MUST have a predatory element (stealing money, phishing credentials, fake giveaways, fake job onboarding, impersonation, or fake payment requests). Err on the side of "allow" if the text reads like a real human conversation.

IMPORTANT DECISION RULES:
- Unexpected texts about loans, package delivery, tolls, tax refunds, benefits, or reward points are suspicious by default unless the text clearly shows the user initiated the request.
- Any request to buy gift cards, send the numbers/PIN, move money to a "safe" place, buy gold, or deposit crypto to unlock earnings should be treated as scam behavior.
- A thin military wrapper does not make a scam legitimate. If the post mentions PCS/deployment/VA but still asks for gift cards, crypto, SSN, bank login, or off-platform payment, treat it as predatory.

Return ONLY valid JSON in this exact schema. You MUST provide your reasoning first before the boolean classification:
{{
  "reasoning": "Step-by-step thought process analyzing the red and green flags...",
  "is_scam": boolean,
  "confidence": number,
  "indicators": [string],
  "recommended_action": "allow" | "flag_for_review" | "block"
}}

Comment: "{comment_text}"
"""

DISTILLATION_PROMPT_TEMPLATE = """
You are an expert psychological linguist acting as a teacher model for high-quality soft-label generation.
Analyze the following comment from a military family forum and infer the most human-like emotional reading of the text independently.

CRITICAL SCORING INSTRUCTIONS (PREVENT LABEL NOISE AND PRIOR BIAS):
1. OUTPUT CALIBRATED SOFT LABELS: Return a score between 0.0 and 1.0 for every emotion. Reserve scores >= 0.50 for strong evidence.
2. INFER FROM THIS TEXT ONLY: Do not rely on class frequency, prior dataset imbalance, or typical forum patterns. Treat each text as its own case.
3. DEFAULT TO CALM / INFORMATIONAL WHEN APPROPRIATE: Many forum comments are informational, logistical, or light conversation. In those cases, keep most emotions near 0.0 and score "neutral" or "curiosity" appropriately.
4. AVOID OVER-LABELING: Usually only 1 to 3 emotions should have meaningful non-zero scores. Do not make every text emotional.
5. RARE EMOTIONS REQUIRE EVIDENCE, NOT FORCE-ZERO: Emotions like "grief", "relief", and "pride" are uncommon, but if there is partial evidence you may use a low soft score such as 0.05 to 0.25 instead of forcing 0.0.
6. USE CONTRASTIVE REASONING: Distinguish sadness vs grief, gratitude vs relief, and neutral vs curiosity before assigning scores.
7. GRATITUDE SHOULD NOT DOMINATE EVERYTHING: Use "gratitude" only when the text clearly expresses thanks or appreciation.
8. If the text is ambiguous, prefer lower scores rather than confident hard positives.
9. Use deep reasoning silently before scoring. Do not anchor to student predictions or class frequency.
10. Reason silently and return only the final JSON.

Return ONLY valid JSON in this exact schema. You MUST provide your reasoning first:
{{
  "reasoning": "Brief explanation of the primary emotions detected...",
  "admiration": float, "amusement": float, "anger": float, "annoyance": float, "approval": float,
  "caring": float, "confusion": float, "curiosity": float, "desire": float, "disappointment": float,
  "disapproval": float, "disgust": float, "embarrassment": float, "excitement": float, "fear": float,
  "gratitude": float, "grief": float, "joy": float, "love": float, "nervousness": float,
  "optimism": float, "pride": float, "realization": float, "relief": float, "remorse": float,
  "sadness": float, "surprise": float, "neutral": float
}}

Comment: "{comment_text}"
"""

BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE = """
You are a Blue Star Families Community Context Reviewer trained on military and veteran family patterns.

Your role: Re-evaluate whether a flagged comment is actually a scam or a legitimate military-family discussion that was over-flagged by generic heuristics.

MILITARY CONTEXT PATTERNS (NOT SCAMS — These are normal in military communities):
- PCS (Permanent Change of Station): Moving info, local recommendations, questions about new duty stations
- Deployment cycles: Family communication, schedule questions, coping resources
- EFMP (Exceptional Family Member Program): Housing, schooling, medical accommodations
- VA benefits/claims: Questions about disability, healthcare, survivor benefits, appeals
- Military spouse/family dynamics: Relocation stress, childcare coordination, family strain management
- Off-platform contact for peer support: Military families often exchange emails/phones for buddy systems
- Older communication style: Formal signatures with name, phone, email on posts (common in older generations)
- Military business/services: Moving companies, real estate, military-family consulting (legitimate)
- Legitimate jobs/services in community context usually include concrete details, public websites, or normal referrals — not secrecy, deposits, or strange payment requests.

RED FLAGS THAT OVERRIDE CONTEXT (Still predatory even in military context):
- "Money-flip" schemes, guaranteed returns, investment solicitations
- Fake giveaways ("You've won...", "Claim your...", no legitimate sponsor)
- Phishing links disguised as official military/VA sites
- Romance scams targeting military families specifically
- Requests for gift card numbers, crypto deposits, SSN, bank login, verification codes, redelivery fees, "safe account" transfers, or gold purchases
- Fake recruiter / task scams using WhatsApp, Telegram, "optimization" work, or pay-to-unlock earnings

CRITICAL OVERRIDE RULE:
Mentioning PCS, deployment, EFMP, VA, or military spouse life is NOT enough for context_pass if the post still asks for money movement, payment codes, sensitive personal information, or urgent off-platform follow-up.

TASK: 
Given a flagged comment, determine:
1. Is this a normal military-family discussion that was over-flagged? → OUTPUT: "context_pass"
2. Is this a legitimate scam that should stay flagged? → OUTPUT: "context_fail"

REASONING: Explain your assessment in 1-2 sentences focusing on military context.
OUTPUT FORMAT:
{{
  "reasoning": "Brief military-context assessment",
  "context_verdict": "context_pass" | "context_fail",
  "confidence": 0.0-1.0,
  "bsf_pattern": "military pattern name or 'predatory'"
}}

Comment: "{comment_text}"
"""

# ==========================================
# --- UTILITY: DATA CLEANING & PARSING ---
# ==========================================
def clean_text(text):
    """Strips HTML tags and normalizes whitespace for better AI tokenization."""
    if not isinstance(text, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode HTML entities (e.g., &amp; -> &)
    text = html.unescape(text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_json_from_text(text):
    """Strips markdown formatting (e.g., ```json ... ```) from LLM outputs to prevent parsing crashes."""
    text = text.strip()
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1)
    start = text.find('{')
    end = text.rfind('}')
    if start != -1 and end != -1 and end > start:
        return text[start:end+1]
    return text

def has_false_positive_conflict_signal(text):
    text = clean_text(text).lower()
    if not text:
        return False
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in FALSE_POSITIVE_CONFLICT_PATTERNS)

def compute_bucket_targets(total_size, sampling_plan):
    exact_targets = {bucket: total_size * ratio for bucket, ratio in sampling_plan}
    bucket_targets = {bucket: int(np.floor(value)) for bucket, value in exact_targets.items()}
    remainder = total_size - sum(bucket_targets.values())
    if remainder > 0:
        ranked_remainders = sorted(
            sampling_plan,
            key=lambda item: (exact_targets[item[0]] - bucket_targets[item[0]], item[1]),
            reverse=True
        )
        for bucket, _ in ranked_remainders[:remainder]:
            bucket_targets[bucket] += 1
    return bucket_targets

def process_in_parallel_batched(func, items, progress_bar=None, status_text=None, status_template="", max_workers=10, batch_size=PRODUCTION_BATCH_SIZE):
    results = {}
    total_items = len(items)
    if total_items == 0:
        return results

    completed_total = 0
    for batch_start in range(0, total_items, batch_size):
        batch_items = items[batch_start:batch_start + batch_size]

        def _batch_progress(completed_in_batch, batch_total):
            overall_completed = completed_total + completed_in_batch
            if progress_bar and status_text:
                progress_bar.progress(min(overall_completed / total_items, 1.0))
                status_text.markdown(status_template.format(completed=overall_completed, total=total_items))

        batch_results = process_in_parallel(
            func,
            batch_items,
            progress_bar=None,
            status_text=None,
            status_template=status_template,
            max_workers=max_workers,
            progress_callback=_batch_progress
        )
        for local_idx, value in batch_results.items():
            results[batch_start + local_idx] = value
        completed_total += len(batch_items)

    if progress_bar and status_text:
        progress_bar.progress(1.0)
        status_text.markdown(status_template.format(completed=total_items, total=total_items))
    return results

# ==========================================
# --- AUTHENTICATION SYSTEM ---
# ==========================================
def get_configured_users():
    try:
        users = st.secrets["users"]
    except KeyError:
        st.error(":material/error: Missing `[users]` section in `.streamlit/secrets.toml`.")
        return None

    if not users:
        st.error(":material/error: No login accounts were found in the `[users]` section of `.streamlit/secrets.toml`.")
        return None

    return {str(email).lower().strip(): str(secret) for email, secret in users.items()}

def verify_user(email, password):
    users = get_configured_users()
    if users is None:
        return None

    email = email.lower().strip()
    return users.get(email) == password


def clear_authenticated_session():
    """Remove all session values so logout cannot retain identity or analysis data."""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.session_state['authenticated'] = False

@st.cache_data
def get_image_as_base64(file):
    try:
        with open(file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except:
        return ""

def render_login_page():
    logo_base64 = get_image_as_base64("logo.png")
    st.markdown(f"""
    <style>
        [data-testid="collapsedControl"] {{ display: none; }}
        [data-testid="stSidebar"] {{ display: none; }}
        [data-testid="stHeader"] {{ display: none; }}
        .stApp {{ background-color: #f0f2f6; background-image: url("data:image/png;base64,{logo_base64}"); background-size: 400px; background-position: center; background-repeat: no-repeat; background-attachment: fixed; }}
        .stApp::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255, 255, 255, 0.92); z-index: -1; }}
        .login-card {{ background: white; padding: 40px; border-radius: 15px; box-shadow: 0 8px 24px rgba(0,0,0,0.1); max-width: 450px; margin: 80px auto; text-align: center; border-top: 5px solid #00387b; }}
        .login-logo {{ width: 150px; margin-bottom: 20px; }}
        .login-title {{ color: #00387b; font-weight: 700; font-size: 24px; margin-bottom: 5px; }}
        .login-subtitle {{ color: #666; font-size: 14px; margin-bottom: 30px; }}
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown(f"""
        <div class="login-card">
            <img src="data:image/png;base64,{logo_base64}" class="login-logo">
            <div class="login-title">Community Wellness AI</div>
            <div class="login-subtitle">Secure Enterprise Access</div>
        </div>
        """, unsafe_allow_html=True)
        st.write("") 
        email = st.text_input("Email Address", placeholder="name@bluestarfam.org")
        password = st.text_input("Password", type="password", placeholder="••••••••")
        if st.button("Secure Log In", type="primary", use_container_width=True):
            auth_result = verify_user(email, password)
            if auth_result is True:
                st.session_state['authenticated'] = True
                st.session_state['user_email'] = email
                st.rerun()
            elif auth_result is False:
                st.error(":material/cancel: Invalid email or password. Contact administrator for access.")

# ==========================================
# --- CORE APP LOGIC & MODELS ---
# ==========================================

@st.cache_resource
def get_snowflake_engine():
    try:
        from cryptography.hazmat.primitives import serialization
        from sqlalchemy import create_engine

        passphrase = st.secrets.snowflake.get("private_key_passphrase")
        p_key_pem = st.secrets.snowflake.private_key
        
        private_key = serialization.load_pem_private_key(
            p_key_pem.encode(),
            password=passphrase.encode() if passphrase else None,
        )

        pkb = private_key.private_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )

        engine = create_engine(
            "snowflake://{user}@{account}/{database}/{schema}?warehouse={warehouse}".format(
                user=st.secrets.snowflake.user,
                account=st.secrets.snowflake.account,
                database=st.secrets.snowflake.database,
                schema=st.secrets.snowflake.schema,
                warehouse=st.secrets.snowflake.warehouse,
            ),
            connect_args={
                'private_key': pkb,
                'client_session_keep_alive': True,
            }
        )
        
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        
        return engine, True
        
    except Exception as e:
        print(f"Snowflake connection error: {e}", file=sys.stderr)
        st.error(":material/error: **Database Connection Failed**")
        st.error("Please verify the database configuration or contact an administrator.")
        return None, False

class RobustEmotionClassifier:
    """Fallback classifier that bypasses transformers.pipeline import issues on some runtimes."""

    def __init__(self, model_id, device_hint):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_id)

        if device_hint == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif isinstance(device_hint, int) and device_hint >= 0 and torch.cuda.is_available():
            self._device = torch.device(f"cuda:{device_hint}")
        else:
            self._device = torch.device("cpu")

        self._model.to(self._device)
        self._model.eval()
        self._id2label = dict(getattr(self._model.config, "id2label", {}))

    def __call__(self, texts, batch_size=16):
        texts = [texts] if isinstance(texts, str) else list(texts)
        outputs = []

        with self._torch.inference_mode():
            for start in range(0, len(texts), max(1, int(batch_size))):
                batch = texts[start:start + max(1, int(batch_size))]
                encoded = self._tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt"
                )
                encoded = {k: v.to(self._device) for k, v in encoded.items()}
                logits = self._model(**encoded).logits

                # GoEmotions is multi-label, so sigmoid calibration is appropriate.
                probs = self._torch.sigmoid(logits).detach().cpu().tolist()

                for row in probs:
                    row_scores = []
                    for idx, score in enumerate(row):
                        label = str(self._id2label.get(idx, idx)).lower()
                        row_scores.append({"label": label, "score": float(score)})
                    outputs.append(row_scores)

        return outputs


class HubAPIEmotionClassifier:
    """Final fallback classifier using Hugging Face Inference API."""

    def __init__(self, model_id):
        self.model_id = model_id
        self.baseline_fallback_count = 0
        self.last_error = None
        token = None
        try:
            token = st.secrets.get("huggingface", {}).get("api_token")
        except Exception:
            token = None
        self.client = InferenceClient(model=model_id, token=token)

    @staticmethod
    def _baseline_row_scores(text: str):
        """Conservative fallback when hosted inference is unavailable."""
        scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
        # Keep fallback neutral-dominant and avoid synthetic distress spikes that
        # can create false severe candidates during transient HF API failures.
        scores['neutral'] = 0.85
        scores['curiosity'] = 0.08
        scores['approval'] = 0.05

        return [{"label": label, "score": float(value)} for label, value in scores.items()]

    @staticmethod
    def _normalize_api_result(result):
        entries = []
        if result is None:
            return entries

        if isinstance(result, dict):
            entries = [result]
        elif isinstance(result, list):
            # Handle list[dict], list[obj], and nested list formats
            if result and isinstance(result[0], list):
                for nested in result:
                    if isinstance(nested, list):
                        entries.extend(nested)
                    else:
                        entries.append(nested)
            else:
                entries = result
        else:
            entries = [result]

        row_scores = []
        for item in entries:
            if isinstance(item, dict):
                label = str(item.get("label", "")).lower()
                score = float(item.get("score", 0.0) or 0.0)
            else:
                label = str(getattr(item, "label", "")).lower()
                score = float(getattr(item, "score", 0.0) or 0.0)
            if label:
                row_scores.append({"label": label, "score": score})
        return row_scores

    def __call__(self, texts, batch_size=16):
        texts = [texts] if isinstance(texts, str) else list(texts)
        outputs = []

        for text in texts:
            try:
                result = None
                try:
                    result = self.client.text_classification(text, top_k=len(EMOTION_LABELS))
                except Exception as e_topk:
                    self.last_error = f"top_k call: {e_topk!r}"
                    result = self.client.text_classification(text)

                row_scores = self._normalize_api_result(result)
                if not row_scores:
                    self.baseline_fallback_count += 1
                    self.last_error = self.last_error or "empty API result"
                    row_scores = self._baseline_row_scores(text)
                outputs.append(row_scores)
            except Exception as e:
                self.baseline_fallback_count += 1
                self.last_error = repr(e)
                outputs.append(self._baseline_row_scores(text))

        return outputs


@st.cache_resource(max_entries=1)
def load_emotion_classifier(model_id="SamLowe/roberta-base-go_emotions"):
    # Auto-detect hardware acceleration (CUDA for NVIDIA, MPS for Apple Silicon)
    device, _ = detect_acceleration_device()
    load_diag = {"pipeline_error": None, "robust_error": None, "api_fallback_error": None, "path": None}

    try:
        from transformers import pipeline
        classifier = pipeline("text-classification", model=model_id, top_k=None, truncation=True, max_length=512, device=device)
        load_diag["path"] = "transformers.pipeline"
        st.session_state["emotion_classifier_diag"] = load_diag
        return classifier, True
    except Exception as pipeline_error:
        load_diag["pipeline_error"] = repr(pipeline_error)
        try:
            fallback_classifier = RobustEmotionClassifier(model_id, device)
            if 'hf_pipeline_fallback_notified' not in st.session_state:
                st.warning(f"HuggingFace pipeline import failed; using robust fallback classifier. Details: {pipeline_error}")
                st.session_state['hf_pipeline_fallback_notified'] = True
            load_diag["path"] = "RobustEmotionClassifier"
            st.session_state["emotion_classifier_diag"] = load_diag
            return fallback_classifier, True
        except Exception as fallback_error:
            load_diag["robust_error"] = repr(fallback_error)
            try:
                api_fallback_classifier = HubAPIEmotionClassifier(model_id)
                load_diag["path"] = "HubAPIEmotionClassifier"
                st.session_state["emotion_classifier_diag"] = load_diag
                return api_fallback_classifier, True
            except Exception as api_fallback_error:
                load_diag["api_fallback_error"] = repr(api_fallback_error)
                load_diag["path"] = "none"
                st.session_state["emotion_classifier_diag"] = load_diag
                st.error(
                    "HuggingFace model loading error: "
                    f"pipeline={pipeline_error} | fallback={fallback_error} | api_fallback={api_fallback_error}"
                )
                return None, False

@st.cache_resource
def load_gemini_model(model_name=FAST_REVIEW_MODEL_NAME):
    try:
        api_key = st.secrets.gemini.api_key
        backend = ensure_gemini_backend_loaded()
        if backend == "unavailable":
            raise RuntimeError("No Gemini SDK available. Install google-genai (preferred) or google-generativeai (legacy).")
        model = GeminiModelAdapter(backend, model_name, api_key)
        return model, True
    except Exception as e:
        print(f"Gemini model loading error: {e}")
        return None, False

def get_teacher_generation_config():
    return build_generation_config(response_mime_type="application/json", temperature=0.1)

class AnalysisProgressUI:
    def __init__(self, step_labels=None):
        self.step_labels = step_labels or ANALYSIS_STEP_LABELS
        self.step_fraction = [0.0 for _ in self.step_labels]
        self.step_complete = [False for _ in self.step_labels]
        self.container = st.container()
        with self.container:
            st.markdown("### 🚦 Pipeline Execution Progress")
            self.overall_status = st.empty()
            self.overall_bar = st.progress(0.0)
            self.step_bars = []
            self.step_status = []
            for idx, label in enumerate(self.step_labels, start=1):
                st.markdown(f"**Step {idx}/{len(self.step_labels)}: {label}**")
                bar = st.progress(0.0)
                status = st.empty()
                status.caption("Pending")
                self.step_bars.append(bar)
                self.step_status.append(status)
        self.set_overall_message("Initializing inference pipeline...")

    def _refresh_overall(self):
        total = max(len(self.step_labels), 1)
        overall_fraction = sum(self.step_fraction) / total
        self.overall_bar.progress(min(max(overall_fraction, 0.0), 1.0))

    def set_overall_message(self, message: str):
        self.overall_status.markdown(message)
        self._refresh_overall()

    def mark_step_started(self, step_index: int, detail: str):
        idx = step_index - 1
        self.step_fraction[idx] = max(self.step_fraction[idx], 0.01)
        self.step_bars[idx].progress(self.step_fraction[idx])
        self.step_status[idx].caption(detail)
        self._refresh_overall()

    def update_step(self, step_index: int, completed: int, total: int, detail: str):
        idx = step_index - 1
        fraction = float(completed) / max(int(total), 1)
        fraction = min(max(fraction, 0.0), 1.0)
        self.step_fraction[idx] = max(self.step_fraction[idx], fraction)
        self.step_complete[idx] = fraction >= 0.999
        self.step_bars[idx].progress(self.step_fraction[idx])
        self.step_status[idx].caption(detail)
        self._refresh_overall()

    def complete_step(self, step_index: int, detail: str):
        idx = step_index - 1
        self.step_fraction[idx] = 1.0
        self.step_complete[idx] = True
        self.step_bars[idx].progress(1.0)
        self.step_status[idx].caption(detail)
        self._refresh_overall()

    def skip_step(self, step_index: int, detail: str):
        self.complete_step(step_index, detail)

    def finalize(self, message: str = "✅ Pipeline execution complete!"):
        self.step_fraction = [1.0 for _ in self.step_labels]
        self.step_complete = [True for _ in self.step_labels]
        self.overall_bar.progress(1.0)
        self.overall_status.success(message)

def validate_threshold_map(raw_thresholds):
    thresholds = {}
    for emotion in EMOTION_LABELS:
        default_value = DEFAULT_LABEL_THRESHOLDS.get(emotion, 0.4)
        try:
            value = float(raw_thresholds.get(emotion, default_value))
        except Exception:
            value = default_value
        thresholds[emotion] = round(min(max(value, 0.01), 0.95), 2)
    return thresholds

def generate_with_retry(model, prompt, generation_config=None, max_retries=3):
    for attempt in range(max_retries):
        try:
            if generation_config:
                return model.generate_content(prompt, generation_config=generation_config)
            return model.generate_content(prompt)
        except Exception as e:
            # Catch AI Safety blocks gracefully
            if "StopCandidateException" in str(type(e)) or "safety" in str(e).lower():
                return type('obj', (object,), {'text': '{"reasoning": "Blocked by AI Safety Filter", "is_scam": false, "confidence": 0.9, "recommended_action": "flag_for_review", "indicators": []}'})()
            if attempt == max_retries - 1:
                raise e
            # Added Jitter to prevent Thundering Herd API crashes
            time.sleep((2 ** attempt) + random.uniform(0.1, 1.5))

def process_in_parallel(func, items, progress_bar=None, status_text=None, status_template="", max_workers=10, progress_callback=None, min_ui_interval=0.15):
    results = {}
    total_items = len(items)
    if total_items == 0:
        return results
    last_ui_update = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(func, item): idx for idx, item in enumerate(items)}
        completed = 0
        for future in concurrent.futures.as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                results[idx] = f"ERROR: {e}"

            completed += 1
            current_time = time.time()
            should_update = (current_time - last_ui_update > min_ui_interval) or completed == total_items
            if should_update:
                if progress_callback is not None:
                    progress_callback(completed, total_items)
                elif progress_bar and status_text:
                    progress_bar.progress(completed / total_items)
                    status_text.markdown(status_template.format(completed=completed, total=total_items))
                last_ui_update = current_time
    return results

def get_week_dates(year, week_number):
    start_of_week = datetime.strptime(f'{year}-W{week_number:02d}-1', "%G-W%V-%u").date()
    end_of_week = start_of_week + timedelta(days=6)
    return start_of_week, end_of_week


def empty_weekly_data_frame():
    return pd.DataFrame(columns=WEEKLY_DATA_COLUMNS)


def load_scam_detection_window(anchor_week: int, anchor_year: int, window_weeks: int = 4):
    """
    Build a rolling dataframe for scam detection: selected week + prior N-1 weeks.
    Returns (combined_df, week_labels) where week_labels are newest -> oldest.
    """
    frames = []
    week_labels = []

    try:
        anchor_date = datetime.strptime(f'{anchor_year}-W{anchor_week:02d}-1', "%G-W%V-%u").date()
    except ValueError:
        return empty_weekly_data_frame(), week_labels

    for offset in range(max(1, int(window_weeks))):
        target_date = anchor_date - timedelta(weeks=offset)
        y, w, _ = target_date.isocalendar()
        week_label = f"W{w}/{y}"
        week_df = load_data_for_week(w, y, show_progress=False)
        if week_df is None or week_df.empty:
            continue

        week_df = week_df.copy()
        week_df["scam_scan_week"] = week_label
        week_df["scam_scan_offset"] = int(offset)
        frames.append(week_df)
        week_labels.append(week_label)

    if not frames:
        return empty_weekly_data_frame(), week_labels

    return pd.concat(frames, ignore_index=True), week_labels

@st.cache_data(ttl=3600)
def load_data_for_week(week, year, show_progress=False):
    from snowflake.connector.errors import ProgrammingError
    status_container = st.empty() if show_progress else None

    for attempt in range(2):
        try:
            engine, snowflake_available = get_snowflake_engine()
            if not snowflake_available:
                if show_progress: st.error("Snowflake connection not available.")
                else: print("Snowflake connection not available.", file=sys.stderr)
                return empty_weekly_data_frame()

            if status_container: status_container.info(f":material/sync: Querying Snowflake for Posts & Comments... (Attempt {attempt + 1})")
            
            start_date, end_date = get_week_dates(year, week)
            
            query = text("""
                WITH Weekly_Posts AS (
                    SELECT 
                        'Original Post' AS activity_type,
                        Forum_Created_At AS activity_date,
                        CONCAT(IFNULL(Forum_Title, ''), ' - ', IFNULL(Forum_Content, '')) AS content,
                        community_name,
                        commenter_name, 
                        commenter_email
                    FROM BSF01_DEV.BSF01_NEIGHBORHOOD.VW_FORUM_DISCUSSIONS_MERGE
                    WHERE TO_DATE(Forum_Created_At) BETWEEN :start_date AND :end_date
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
                    WHERE TO_DATE(Discussion_Created_At) BETWEEN :start_date AND :end_date
                      AND discussion_id IS NOT NULL
                      AND Discussion_Content IS NOT NULL
                      AND TRIM(Discussion_Content) != ''
                    QUALIFY ROW_NUMBER() OVER (PARTITION BY discussion_id ORDER BY Discussion_Created_At DESC) = 1
                )
                
                SELECT * FROM Weekly_Posts
                UNION ALL
                SELECT * FROM Weekly_Comments
            """)
            
            df = pd.read_sql(query, engine, params={"start_date": start_date, "end_date": end_date})
            
            if not df.empty and 'content' in df.columns:
                df['content'] = df['content'].apply(clean_text)
            
            if status_container: 
                post_count = len(df[df['activity_type'] == 'Original Post'])
                comment_count = len(df[df['activity_type'] == 'Comment'])
                status_container.success(f":material/check_circle: Found {len(df)} total interactions ({post_count} Posts, {comment_count} Comments) for Week {week}, {year}")
            break

        except Exception as e:
            is_token_error = isinstance(getattr(e, 'orig', None), ProgrammingError) and "Authentication token has expired" in str(e.orig)
            if is_token_error and attempt == 0:
                if status_container: st.warning("Session expired. Automatically reconnecting...")
                get_snowflake_engine.clear()
                continue
            else:
                print(f"Snowflake query error: {e}", file=sys.stderr)
                if status_container:
                    st.error("The database query failed. Please try again or contact an administrator.")
                return empty_weekly_data_frame()

    if df.empty:
        if show_progress:
            st.error("No data found for this week.")
        else:
            print(f"No data found for Week {week}, {year}.", file=sys.stderr)
        return empty_weekly_data_frame()

    if status_container: status_container.empty()
    return df

def get_emotion_categories():
    return list(EMOTION_LABELS)

def get_label_thresholds():
    session_thresholds = st.session_state.get('active_label_thresholds') if hasattr(st, 'session_state') else None
    if isinstance(session_thresholds, dict) and session_thresholds:
        return validate_threshold_map(session_thresholds)
    return dict(DEFAULT_LABEL_THRESHOLDS)

def normalize_emotion_scores(raw_scores):
    normalized = {}
    for emotion in EMOTION_LABELS:
        try:
            value = float(raw_scores.get(emotion, 0.0))
        except Exception:
            value = 0.0
        normalized[emotion] = max(0.0, min(1.0, value))
    return normalized

def normalize_training_text(text):
    cleaned = clean_text(text)
    return re.sub(r'\s+', ' ', cleaned).strip().lower()

def build_text_hash(text):
    return hashlib.sha256(normalize_training_text(text).encode('utf-8')).hexdigest()

def is_viable_training_text(text):
    cleaned = clean_text(text)
    if not cleaned:
        return False
    alpha_count = len(re.findall(r'[A-Za-z]', cleaned))
    if len(cleaned) >= 15 and alpha_count >= 3:
        return True
    short_signal = bool(re.search(r'\b(help|need|lost|stressed|worried|scared|thanks|thank you|sorry|urgent)\b', cleaned, re.IGNORECASE))
    return len(cleaned) >= 6 and alpha_count >= 2 and short_signal

def get_active_labels_from_scores(scores, thresholds=None):
    thresholds = thresholds or get_label_thresholds()
    active = [emotion for emotion in EMOTION_LABELS if scores.get(emotion, 0.0) >= thresholds.get(emotion, 0.4)]
    if active:
        return active
    top_emotion = max(EMOTION_LABELS, key=lambda emotion: scores.get(emotion, 0.0))
    return [top_emotion] if scores.get(top_emotion, 0.0) > 0 else ['neutral']

def get_rare_hint_emotions(text):
    normalized = normalize_training_text(text)
    matched = []
    for emotion, patterns in RARE_EMOTION_HINTS.items():
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in patterns):
            matched.append(emotion)
    return matched

def get_support_need_hits(text):
    normalized = normalize_training_text(text)
    return [pattern for pattern in SUPPORT_NEED_PATTERNS if re.search(pattern, normalized, re.IGNORECASE)]

def has_support_need_signal(text):
    return bool(get_support_need_hits(text))


def normalize_identity_value(value, fallback="N/A"):
    """Normalize nullable/NaN identity fields for stable UI display and downstream logs."""
    try:
        if value is None or pd.isna(value):
            return fallback
    except Exception:
        pass

    text_value = str(value).strip()
    if not text_value or text_value.lower() in {"nan", "none", "null", "nat"}:
        return fallback
    return text_value


def is_staff_or_admin_comment(commenter_name, commenter_email):
    name = normalize_identity_value(commenter_name, fallback="").lower()
    email = normalize_identity_value(commenter_email, fallback="").lower()
    if "@bluestarfam.org" in email:
        return True
    return any(token in name for token in ["admin", "administrator", "blue star", "bsf", "moderator"])

def compute_severe_candidate_details(score_row, text, review_thresholds=None):
    # Always resolve active_thresh so it is available throughout the function
    active_thresh = get_label_thresholds()
    # Dynamically scale triage thresholds based on the active model's calibration
    if not review_thresholds:
        # Triage band is 75% of the standard threshold to ensure high recall without flooding Gemini
        review_thresholds = {emo: max(0.05, active_thresh.get(emo, 0.4) * 0.75) for emo in EMOTION_LABELS}
        
    scores = normalize_emotion_scores(score_row)
    rare_hints = set(get_rare_hint_emotions(text))
    support_hits = get_support_need_hits(text)
    reason_flags = []

    # Only sum emotions that have a real signal (> 0.10) to ignore LoRA baseline noise
    severe_components = [emotion for emotion in SEVERE_SUM_EMOTIONS if scores.get(emotion, 0.0) > 0.10]
    severe_sum = sum(scores.get(emotion, 0.0) for emotion in severe_components)
    severe_component_count = len(severe_components)
    severe_avg = (severe_sum / severe_component_count) if severe_component_count else 0.0
    severe_spike = max(scores.get(emotion, 0.0) for emotion in SERIOUS_NEGATIVE_EMOTIONS)
    dominant_emotion = max(SERIOUS_NEGATIVE_EMOTIONS + ['confusion'], key=lambda emotion: scores.get(emotion, 0.0))
    dominant_score = scores.get(dominant_emotion, 0.0)

    # Emotion review bands: score alone is sufficient to flag for Gemini review.
    # rare_hints lowers the effective threshold (higher confidence), but is NOT
    # a mandatory gate — removing it was the primary cause of missed detections.
    for emotion in ['grief', 'fear', 'remorse', 'nervousness']:
        base_thresh = SEVERE_REVIEW_THRESHOLDS.get(emotion, review_thresholds.get(emotion, 0.20))
        # If a rare lexical hint corroborates the model signal, accept a lower score.
        effective_thresh = base_thresh * 0.65 if emotion in rare_hints else base_thresh
        if scores.get(emotion, 0.0) >= effective_thresh:
            reason_flags.append(f'{emotion}_review_band')
            
    if scores.get('embarrassment', 0.0) >= review_thresholds.get('embarrassment', 0.15): 
        reason_flags.append('embarrassment_review_band')
        
    if scores.get('sadness', 0.0) >= review_thresholds.get('sadness', 0.30): 
        reason_flags.append('sadness_above_review_band')
        
    if scores.get('disappointment', 0.0) >= review_thresholds.get('disappointment', 0.25) and scores.get('sadness', 0.0) >= active_thresh.get('sadness', 0.40) * 0.6: 
        reason_flags.append('sadness_disappointment_cluster')
        
    if scores.get('fear', 0.0) >= active_thresh.get('fear', 0.40) * 0.5 and scores.get('nervousness', 0.0) >= active_thresh.get('nervousness', 0.25) * 0.5: 
        reason_flags.append('fear_nervousness_cluster')
        
    # Lower severe_sum floor from 0.65 → 0.45 to catch multi-emotion distress clusters
    # that individually stay below single-emotion thresholds.
    dynamic_sum_threshold = sum(active_thresh.get(e, 0.4) for e in SEVERE_SUM_EMOTIONS) * 0.50
    if severe_sum >= max(0.45, dynamic_sum_threshold): 
        reason_flags.append('severe_sum_high')

    # Explicit severe guardrail requested by operations:
    # escalate when cumulative severe load is high OR when one severe signal spikes.
    if severe_sum >= SEVERE_SUM_HARD_THRESHOLD or severe_spike >= SEVERE_SPIKE_HARD_THRESHOLD:
        reason_flags.append('severe_hard_guardrail')

    contextual_distress = max(scores.get(emotion, 0.0) for emotion in SUPPORT_NEED_CONTEXT_EMOTIONS)
    # CALIBRATION POINT 3: Relaxed contextual-distress gate
    # Lower multiplier from 0.30 to 0.20 to catch indirect/implicit hardship signals
    # (e.g., "thank you for support" when member is clearly struggling)
    contextual_floor = active_thresh.get(dominant_emotion, 0.4) * 0.20
    if support_hits and contextual_distress >= contextual_floor:
        reason_flags.append('support_need_signal')
    elif support_hits and scores.get('grief', 0.0) >= 0.08:
        # Grief signal with a support-need keyword doesn't need the rare_hints gate
        reason_flags.append('support_need_with_grief_hint')

    # CALIBRATION POINT 4: Lexical hardship override
    # Explicit distress/vulnerability phrases bypass emotion-score gates
    # Catches: "I'm homeless", "I'm disabled", "Can you help me?" even with moderate emotion scores
    hardship_override_patterns = [
        r"i'?m homeless", r"i'?m disabled", r"can you help me", r"can y'?all help",
        r"i need help", r"we'?re struggling", r"my child is autistic",
        r"is anyone there", r"anyone[,\s]+please", r"really need"
    ]
    has_hardship_trigger = any(re.search(pattern, text.lower()) for pattern in hardship_override_patterns)
    if has_hardship_trigger and contextual_distress >= 0.08:
        # Only apply if some distress signal present (avoid flagging simple statements)
        if 'lexical_hardship_override' not in reason_flags:
            reason_flags.append('lexical_hardship_override')

    # CALIBRATION POINT 5: Multi-hit context override (resilient framing)
    # Members who describe hardship positively ("grit and grace after job loss") are missed
    # by emotion gates because RoBERTa scores their optimistic tone, not their situation.
    # When 3+ support-need patterns fire AND at least one high-signal hardship keyword
    # is present, trust the situational evidence over the emotional framing.
    _high_signal_hardship = [
        r'\bjob loss\b', r'\blaid off\b', r'\blayoff\b', r'\bcaregiver\b', r'\bcaretaker\b',
        r'\bbeen tough\b', r'\btough time\b', r'\bhard times\b', r'\bmaking ends meet\b',
        r'\bfood insecurity\b', r'\bhomeless\b', r'\bdisabled\b', r'\beviction\b',
        r'\bbehind on bills\b', r'\bfinancial hardship\b', r'\bwidow\b'
    ]
    if len(support_hits) >= 3 and any(re.search(p, text.lower()) for p in _high_signal_hardship):
        if 'multi_hit_context_override' not in reason_flags:
            reason_flags.append('multi_hit_context_override')

    # Precision guardrail: skip low-signal single-emotion worry chatter unless
    # there is stronger severe evidence or concrete support-need context.
    low_signal_flags = {'fear_review_band', 'nervousness_review_band', 'embarrassment_review_band'}
    if reason_flags:
        has_strong_signal = any(
            flag in {
                'severe_sum_high', 'severe_hard_guardrail', 'support_need_signal',
                'support_need_with_grief_hint', 'sadness_above_review_band',
                'grief_review_band', 'remorse_review_band'
            }
            for flag in reason_flags
        )
        if (not has_strong_signal) and set(reason_flags).issubset(low_signal_flags) and not support_hits and severe_component_count <= 1:
            reason_flags = []

    return {
        'candidate': bool(reason_flags),
        'severe_sum': severe_sum,
        'severe_avg': severe_avg,
        'severe_components': severe_components,
        'severe_component_count': severe_component_count,
        'severe_spike': severe_spike,
        'dominant_emotion': dominant_emotion,
        'dominant_score': dominant_score,
        'reason_flags': reason_flags,
        'rare_hints': sorted(rare_hints),
        'support_need_hits': support_hits,
    }

def compute_mild_moderate_details(score_row, text):
    scores = normalize_emotion_scores(score_row)
    severe_details = compute_severe_candidate_details(scores, text)
    if severe_details['candidate']:
        return {
            'candidate': False,
            'negative_sum': severe_details['severe_sum'],
            'negative_avg': severe_details.get('severe_avg', 0.0),
            'negative_components': severe_details.get('severe_components', []),
            'negative_component_count': severe_details.get('severe_component_count', 0),
            'dominant_emotion': severe_details['dominant_emotion'],
            'dominant_score': severe_details['dominant_score'],
            'support_need_hits': severe_details['support_need_hits'],
        }

    negative_emotions = [emotion for emotion in NEGATIVE_EMOTIONS if emotion != 'confusion']
    negative_components = [emotion for emotion in negative_emotions if scores.get(emotion, 0.0) > 0.10]
    negative_component_count = len(negative_components)
    negative_sum = sum(scores.get(emotion, 0.0) for emotion in negative_emotions)
    negative_avg = (negative_sum / negative_component_count) if negative_component_count else 0.0
    dominant_emotion = max(negative_emotions + ['confusion'], key=lambda emotion: scores.get(emotion, 0.0))
    dominant_score = scores.get(dominant_emotion, 0.0)
    support_hits = get_support_need_hits(text)
    contextual_peak = max(scores.get(emotion, 0.0) for emotion in SUPPORT_NEED_CONTEXT_EMOTIONS)

    candidate = (
        (0.20 <= negative_sum < 0.55)
        or (contextual_peak >= 0.18 and negative_sum >= 0.12)
        or (bool(support_hits) and contextual_peak >= 0.12)
    )

    return {
        'candidate': bool(candidate and dominant_score > 0),
        'negative_sum': negative_sum,
        'negative_avg': negative_avg,
        'negative_components': negative_components,
        'negative_component_count': negative_component_count,
        'dominant_emotion': dominant_emotion,
        'dominant_score': dominant_score,
        'support_need_hits': support_hits,
    }

def parse_gemini_boolean_verdict(raw_text):
    if not isinstance(raw_text, str):
        return None
    clean = raw_text.upper().replace("*", "").replace("#", "").replace(" ", "")
    if "VERDICT:TRUE" in clean:
        return True
    if "VERDICT:FALSE" in clean:
        return False
    if "TRUE" in clean and "FALSE" not in clean:
        return True
    if "FALSE" in clean and "TRUE" not in clean:
        return False
    return None

def score_threshold_margin(score_row, thresholds=None):
    thresholds = thresholds or get_label_thresholds()
    return min(abs(float(score_row.get(emotion, 0.0)) - thresholds.get(emotion, 0.4)) for emotion in EMOTION_LABELS)

# --- SLIDING WINDOW INFERENCE ---
def chunk_text_sliding_window(text, chunk_size=300, overlap=50):
    """Splits long text into overlapping chunks to prevent truncation loss."""
    words = text.split()
    if len(words) <= chunk_size: return [text]
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i:i + chunk_size]))
    return chunks

def _run_inference_with_chunking(texts, classifier):
    """Runs inference with sliding window and aggregates max scores."""
    final_rows = []
    for text in texts:
        chunks = chunk_text_sliding_window(text)
        try:
            # Explicitly pass batch_size to the pipeline to utilize GPU
            chunk_results = classifier(chunks, batch_size=16)
            # Aggregate by taking the max score for each emotion across all chunks
            aggregated_scores = {emotion: 0.0 for emotion in EMOTION_LABELS}
            for res in chunk_results:
                res_list = res if isinstance(res, list) else [res]
                for r in res_list:
                    if r['score'] > aggregated_scores.get(r['label'], 0.0):
                        aggregated_scores[r['label']] = r['score']
            final_rows.append(aggregated_scores)
        except Exception:
            final_rows.append({})
    return final_rows

@st.cache_data(ttl=3600, show_spinner=False)
def analyze_emotions_cached(df, model_id="SamLowe/roberta-base-go_emotions"):
    classifier, huggingface_available = load_emotion_classifier(model_id)
    if df.empty or not huggingface_available or 'content' not in df.columns:
        return pd.DataFrame(columns=get_emotion_categories())
    
    texts = df['content'].astype(str).tolist()
    rows = []
    batch_size = 32 
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        rows.extend(_run_inference_with_chunking(batch, classifier))
            
    return pd.DataFrame(rows).fillna(0)

def analyze_emotions_with_ui(df, model_id, progress_ui=None):
    classifier, huggingface_available = load_emotion_classifier(model_id)
    if df.empty or not huggingface_available or 'content' not in df.columns:
        if progress_ui:
            progress_ui.skip_step(1, 'No interactions available for emotion analysis.')
        return pd.DataFrame(columns=get_emotion_categories())

    texts = df['content'].astype(str).tolist()
    rows = []
    batch_size = 32
    total_comments = len(texts)
    
    # Detect hardware acceleration for UI display
    _, device_type = detect_acceleration_device()
    
    if progress_ui:
        progress_ui.mark_step_started(1, f'Running RoBERTa batch inference on {total_comments} interactions using {device_type} acceleration (Sliding Window Chunking enabled)...')

    for i in range(0, total_comments, batch_size):
        batch = texts[i:i+batch_size]
        rows.extend(_run_inference_with_chunking(batch, classifier))

        if progress_ui:
            completed = min(i + batch_size, total_comments)
            progress_ui.update_step(1, completed, total_comments, f'Inference batch {completed}/{total_comments} ({device_type})')

    if progress_ui:
        progress_ui.complete_step(1, f'Emotion analysis complete for {total_comments} interactions using {device_type} acceleration.')
    try:
        bc = getattr(classifier, 'baseline_fallback_count', None)
        if bc is not None:
            st.session_state['emotion_baseline_fallback_count'] = int(bc)
            st.session_state['emotion_baseline_last_error'] = getattr(classifier, 'last_error', None)
    except Exception:
        pass
    return pd.DataFrame(rows).fillna(0)

def compute_health_score(avg):
    categories = get_emotion_categories()
    negative_emotions = categories[:12]
    neg_intensity = avg.reindex(negative_emotions, fill_value=0).sum()
    total_intensity = avg.sum()
    if total_intensity == 0: return None
    negative_intensity_pct = (neg_intensity / total_intensity) * 100
    health_score = 10 * (1 - negative_intensity_pct / 100)
    return round(health_score, 2)

def count_negative_dominant_comments(df_em):
    negative_emotions = get_emotion_categories()[:12]
    all_emotions = get_emotion_categories()
    negative_dominant_count = 0
    for i, row in df_em.iterrows():
        max_emotion = None
        max_intensity = 0
        for emotion in all_emotions:
            intensity = row.get(emotion, 0)
            if intensity > max_intensity:
                max_intensity = intensity
                max_emotion = emotion
        if max_emotion in negative_emotions:
            negative_dominant_count += 1
    return negative_dominant_count

def count_mild_moderate_comments(df_em, df=None):
    mild_moderate_count = 0
    for i, row in df_em.iterrows():
        text_value = ''
        if df is not None and i < len(df):
            text_value = df.iloc[i].get('content', '')
        details = compute_mild_moderate_details(row, text_value)
        if details['candidate']:
            mild_moderate_count += 1
    return mild_moderate_count

def make_summary(score, avg, concerns, count, df_em, df=None, prev_score=None, week=None, year=None, scam_count=0):
    if score is None or avg.empty:
        return "<strong>No data available for this week.</strong>"
    total_comments = count
    negative_emotions = get_emotion_categories()[:12]
    week_info = f"Week {week}, {year}" if week and year else "this week"
    summary = f"<strong>This week's community snapshot of {total_comments} unique interactions ({week_info}):</strong>\n\n"
    
    if prev_score is not None:
        delta_score = round(score - prev_score, 2)
        direction = "📈" if delta_score > 0 else "📉" if delta_score < 0 else "➡️"
        summary += f"• <strong>Health score:</strong> {score}/10 ({direction} {abs(delta_score)} points vs. {prev_score} last week)\n\n"
    else:
        summary += f"• <strong>Health score:</strong> {score}/10\n\n"
    summary += f"• <strong>Fraud / scam flags:</strong> {int(scam_count)} interaction(s) flagged for review\n\n"

    top_neg = avg[negative_emotions][avg[negative_emotions] > 0.01].sort_values(ascending=False).head(3)
    if not top_neg.empty:
        concerns_list =[f"{emotion} {round(avg[emotion]*100, 1)}%" for emotion in top_neg.index]
        summary += f"• <strong>Top concerns:</strong> {', '.join(concerns_list)}\n\n"

    severe_flags = len(concerns)
    mild_moderate_flags = count_mild_moderate_comments(df_em, df)
    summary += f"• <strong>Severe interactions (Gemini Verified):</strong> {severe_flags} items\n\n"
    summary += f"• <strong>Mild–moderate worry interactions:</strong> {mild_moderate_flags} items\n\n"
    negative_dominant = count_negative_dominant_comments(df_em)
    summary += f"• <strong>Interactions where strongest emotion was negative:</strong> {negative_dominant} items\n"
    return summary

SCAM_KEYWORD_TERMS = [
    # Giveaway bait (high signal — always suspicious)
    "giving away", "giveaway", "no cost", "won", "winner", "claim your", "congratulations you",
    # Off-platform routing (specific, not generic "contact me")
    "send me a dm", "dm me",
    # High-risk apps used to route victims off-platform
    "whatsapp", "telegram", "signal", "kik", "snapchat",
    # Payment fraud / financial terms
    "gift card", "wire transfer", "western union", "moneygram", "crypto", "bitcoin", "eth", "usdt",
    "cashapp", "cash app", "venmo", "zelle", "cash flip",
    # Phishing triggers (specific account-takeover language)
    "verify your account", "reset your password",
    # Urgency pressure combined with financial ask
    "act fast", "limited time offer",
    # Financial fraud schemes
    "easy money", "forex", "guaranteed returns", "cash flip",
    # Predatory / explicit scam patterns
    "sugar daddy", "sugar baby", "orphanage",
    # REMOVED (too common in legitimate BSF posts):
    # "free", "kindly", "work from home", "login", "sign in", "urgent", "contact me",
    # "message me", "email me", "direct message", "passive income", "investment",
    # "donate now", "gofundme", "paypal", "apple pay", "google pay", "click the link",
    # "tap the link", "allowance"
]

HIGH_RISK_PAYMENT_TERMS = [
    "gift card", "wire", "western union", "moneygram", "crypto", "bitcoin", "eth", "usdt",
    "cashapp", "cash app", "venmo", "zelle", "paypal", "apple pay", "google pay", "cash flip"
]

OFFPLATFORM_CONTACT_TERMS = [
    # Specific DM-routing terms and high-risk apps only — not generic "contact me" / "email me"
    "send me a dm", "dm me", "whatsapp", "telegram", "signal", "kik", "snapchat"
]

URGENCY_PRESSURE_TERMS = [
    "act fast", "limited time", "urgent", "hurry", "time sensitive", "asap", "quickly",
    "don't miss out", "offer expires", "today only", "immediately", "now", "immediately asap", "right now"
]

INVESTMENT_FRAUD_TERMS = [
    "guaranteed returns", "easy money", "passive income", "guaranteed profit",
    "forex", "trading", "returns", "investment opportunity", "roi", "doubled", "triple"
]

JOB_FRAUD_TERMS = [
    "make money fast", "earn $", "quick cash", "guarantees",
    "pay upfront", "upfront deposit", "upfront payment to start",
    "online assessor for money", "remote work guaranteed income", "daily guaranteed pay"
]

SMISHING_SCAM_TERMS = [
    "delivery fee", "delivery issue", "package on hold", "redelivery", "track your package",
    "reward points", "traffic violation", "toll violation", "tax refund", "hearing notice"
]

LOAN_SCAM_TERMS = [
    "loan application", "loan request", "preapproved", "pre-approved", "claim the loan", "finish your application"
]

ENGAGEMENT_BAIT_TERMS = [
    "reply yes now", "reply yes to claim", "confirm now to unlock", "click yes to win"
]

TASK_SCAM_TERMS = [
    "optimization", "optimizing", "combo task", "combo tasks", "rate products",
    "simple clicks", "unlock earnings", "task platform"
]

IMPERSONATION_SCAM_TERMS = [
    "urgent favor", "your boss", "safe account", "protect your money", "move your money",
    "buy gold", "gift card numbers", "pin on the back", "fraud department", "benefits team", "google certificates", "do not call"
]

PRIZE_FEE_TERMS = [
    "processing fee", "shipping fee", "pay the taxes", "claim your prize", "handling charges"
]

PII_REQUEST_TERMS = [
    "social security", "ssn", "bank account", "banking details", "routing number", "verification code", "login details"
]

DEBT_RELIEF_SCAM_TERMS = [
    "debt relief", "erase your debt", "enrollment fee", "balances will disappear"
]

BENEFIT_IMPOSTOR_TERMS = [
    "claim your tax refund now", "unclaimed benefits claim", "federal mortgage relief", "government va payment pending"
]

SCAM_AWARENESS_SAFE_TERMS = [
    "official app", "official site", "official website", "not random texts", "is this a scam",
    "scam alert", "do not click", "don't click", "beware", "warning"
]

# Generic spam patterns (NOT military-scam-specific) — to improve recall on SMS/email spam
GENERIC_LOTTERY_SPAM = [
    "congratulations", "you won", "claim your prize", "you've won", "congratulations you",
    "tap to claim", "tap here", "free prize", "prize winner", "lucky winner"
]

GENERIC_PHISHING = [
    "verify account", "verify your account", "update payment", "confirm identity",
    "unusual activity detected", "suspicious activity", "click here immediately",
    "click link", "click this link", "unusual access"
]

GENERIC_SMS_SPAM = [
    "message status", "package delivery", "delivery status", "activate now", "click link",
    "confirm delivery", "limited time", "offer expires", "exclusive offer", "tap link"
]

GENERIC_FINANCIAL_SPAM = [
    "free money", "easy cash", "passive income", "guaranteed returns", "make money fast",
    "earn $", "click to earn", "money waiting"
]

SHORTLINK_RE = re.compile(r"\b(bit\.ly|t\.co|tinyurl\.com|goo\.gl|ow\.ly|is\.gd|buff\.ly|rebrand\.ly)\b", re.IGNORECASE)
URL_RE = re.compile(r"\bhttps?://\S+\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?:(?:\+?1\s*[-.(]?)?\s*\d{3}\s*[-.)]?\s*\d{3}\s*[-.]?\s*\d{4})")
LEGACY_SMS_ACTION_RE = re.compile(r"\b(txt|text|reply|call|stop|optout|opt\s*out)\b", re.IGNORECASE)
LEGACY_SMS_PROMO_STRONG_RE = re.compile(r"\b(win|won|winner|prize|claim|ringtone|tones|cash)\b", re.IGNORECASE)
LEGACY_SMS_FREE_RE = re.compile(r"\bfree\b", re.IGNORECASE)
SMS_SHORTCODE_RE = re.compile(r"(?<!\d)\d{5}(?!\d)")
SMS_SHORTCODE_ACTION_RE = re.compile(r"\b(txt|text|reply|call|stop|optout|opt\s*out|send)\b", re.IGNORECASE)
SMS_SHORTCODE_PROMO_RE = re.compile(r"\b(win|won|winner|prize|claim|free|discount|voucher|tone|tones|video|chat|draw|cash)\b", re.IGNORECASE)
ACCOUNT_SECURITY_PHISH_SIGNAL_RE = re.compile(
    r"\b(unauthorized access|failed login|verify your account|security notification|internet banking|account (?:locked|suspended)|confirm identity)\b",
    re.IGNORECASE,
)
ACCOUNT_SECURITY_PHISH_CONTEXT_RE = re.compile(r"\b(bank|account|login|password|security|member)\b", re.IGNORECASE)


def extract_detected_contact_email(text: str) -> str:
    match = EMAIL_RE.search(text or "")
    return normalize_identity_value(match.group(0), fallback="N/A") if match else "N/A"


def _payment_term_present(term: str, text: str) -> bool:
    """Match payment terms with token-aware handling for short symbols like ETH/USDT."""
    term = str(term or "").lower().strip()
    text = str(text or "").lower()
    if not term or not text:
        return False

    # Short alphabetic symbols are prone to substring collisions (e.g., "eth" in "mentioned").
    if term.isalpha() and len(term) <= 4:
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text, re.IGNORECASE) is not None

    return term in text

def _normalize_text_for_scan(text: str) -> str:
    return (text or "").lower()

def scam_heuristic_scan(text: str) -> dict:
    t = _normalize_text_for_scan(text)
    matched =[]
    for term in SCAM_KEYWORD_TERMS:
        if term in t: matched.append(term)
    if EMAIL_RE.search(text or ""): matched.append("email_address")
    if PHONE_RE.search(text or ""): matched.append("phone_number")
    if URL_RE.search(text or ""): matched.append("url")
    if SHORTLINK_RE.search(text or ""): matched.append("shortlink")

    score = 0
    if "email_address" in matched: score += 2   # reduced from 4: email alone is normal in BSF community
    if "phone_number" in matched: score += 2   # reduced from 3: older members sign posts with phone
    if "shortlink" in matched: score += 4       # raised: shortlinks are genuinely suspicious
    if "url" in matched: score += 1             # reduced from 2: URLs are ubiquitous
    score += min(6, len([m for m in matched if m not in ("email_address","phone_number","url","shortlink")]))

    giveaway = any(k in matched for k in ("giving away", "giveaway", "won", "winner", "claim your"))
    # Off-platform: only specific DM-routing terms and high-risk apps, not generic "contact me"
    offplatform = any(k in matched for k in ("send me a dm", "dm me", "whatsapp", "telegram", "signal", "kik", "snapchat"))
    
    # Check directly in text (not just in matched keywords)
    has_payment = any(_payment_term_present(p, t) for p in HIGH_RISK_PAYMENT_TERMS)
    payment_count = sum(1 for p in HIGH_RISK_PAYMENT_TERMS if _payment_term_present(p, t))
    has_predatory = any(k in matched for k in ("sugar daddy", "sugar baby", "orphanage"))
    has_lottery = "lottery" in t
    
    # NEW: Tier 1 improvements for better recall
    has_urgency = any(u in t for u in URGENCY_PRESSURE_TERMS)
    has_investment_fraud = any(inv in t for inv in INVESTMENT_FRAUD_TERMS)
    has_job_fraud = any(job in t for job in JOB_FRAUD_TERMS)
    has_smishing = any(term in t for term in SMISHING_SCAM_TERMS)
    has_loan_scam = any(term in t for term in LOAN_SCAM_TERMS)
    has_engagement_bait = any(term in t for term in ENGAGEMENT_BAIT_TERMS)
    has_task_scam = any(term in t for term in TASK_SCAM_TERMS)
    has_impersonation = any(term in t for term in IMPERSONATION_SCAM_TERMS)
    has_prize_fee = any(term in t for term in PRIZE_FEE_TERMS)
    has_pii_request = any(term in t for term in PII_REQUEST_TERMS)
    has_debt_relief_scam = any(term in t for term in DEBT_RELIEF_SCAM_TERMS)
    has_benefit_impostor = any(term in t for term in BENEFIT_IMPOSTOR_TERMS)
    has_awareness_safe_context = any(term in t for term in SCAM_AWARENESS_SAFE_TERMS)
    
    # Payment routing intent (sending money or upfront payment language)
    send_payment = any(phrase in t for phrase in ("send", "transfer", "pay", "wire", "send me", "upfront"))
    has_link = "url" in matched or "shortlink" in matched
    mentions_gift_card = "gift card" in t or "gift card" in matched
    mentions_gold = "gold" in t
    mentions_withdraw_cash = "withdraw cash" in t
    mentions_pin_codes = " pin" in f" {t}" or "pins" in t or "numbers on the back" in t
    
    if giveaway and offplatform: score += 4          # bait + routing = classic scam combo
    if has_payment and offplatform: score += 3       # payment ask + off-platform routing
    if payment_count >= 2: score += 2                # multiple payment platforms = financial fraud pattern
    if has_predatory and offplatform: score += 4     # predatory targeting + off-platform routing
    if has_predatory and has_payment: score += 4     # predatory targeting + payment ask
    
    # NEW: Urgency + single payment = financial pressure fraud (wire transfer scams)
    if has_urgency and has_payment: score += 3
    
    # NEW: Investment fraud combos
    if has_investment_fraud and (has_payment or offplatform): score += 5
    elif has_investment_fraud and has_urgency: score += 2  # investment scams often use urgency even without payment
    
    # NEW: Job scam combos (work-from-home requests + payment + urgency)
    if has_job_fraud and has_payment and has_urgency: score += 5
    elif has_job_fraud and has_payment: score += 2  # job scams with payment alone are suspicious
    
    # NEW: Send/pay/transfer language + payment method = payment fraud routing
    if send_payment and has_payment: score += 3

    # NEW: delivery/toll/reward/tax smishing patterns — especially when paired with links or urgency
    if has_smishing and has_link: score += 4
    if has_smishing and has_urgency: score += 2

    # NEW: fake loan application follow-ups asking for engagement or private info
    if has_loan_scam and has_payment: score += 4
    if has_loan_scam and (has_engagement_bait or has_pii_request): score += 4

    # NEW: fake recruiter / task scams frequently mix vague jobs with crypto or off-platform routing
    if has_task_scam and (has_payment or offplatform): score += 5
    if has_job_fraud and offplatform and (has_engagement_bait or has_urgency): score += 3
    elif has_job_fraud and offplatform: score += 2

    # NEW: impersonation scams demanding gift cards, safe transfers, gold, or private details
    if has_impersonation and mentions_gift_card: score += 5
    if has_impersonation and (has_payment or has_pii_request or mentions_gold or mentions_withdraw_cash): score += 5
    if has_impersonation and mentions_pin_codes: score += 5

    # NEW: prize / grant / benefit scams that require fees before release
    if has_prize_fee: score += 5

    # NEW: debt-relief and benefits impostor stories often ask for fees or private data with no obvious payment keyword
    if has_debt_relief_scam: score += 5
    if has_benefit_impostor and (has_pii_request or send_payment or offplatform): score += 5

    # GENERIC SPAM PATTERNS (NOT military-scam-specific) — improve recall on SMS/email spam
    has_generic_lottery = any(term in t for term in GENERIC_LOTTERY_SPAM)
    has_generic_phishing = any(term in t for term in GENERIC_PHISHING)
    has_generic_sms_spam = any(term in t for term in GENERIC_SMS_SPAM)
    has_generic_financial = any(term in t for term in GENERIC_FINANCIAL_SPAM)
    has_legacy_sms_action = LEGACY_SMS_ACTION_RE.search(t) is not None
    has_legacy_sms_promo_strong = LEGACY_SMS_PROMO_STRONG_RE.search(t) is not None
    has_legacy_sms_free = LEGACY_SMS_FREE_RE.search(t) is not None
    has_sms_shortcode = SMS_SHORTCODE_RE.search(t) is not None
    has_sms_shortcode_action = SMS_SHORTCODE_ACTION_RE.search(t) is not None
    has_sms_shortcode_promo = SMS_SHORTCODE_PROMO_RE.search(t) is not None
    has_account_security_phish_signal = ACCOUNT_SECURITY_PHISH_SIGNAL_RE.search(t) is not None
    account_security_context_hits = len(ACCOUNT_SECURITY_PHISH_CONTEXT_RE.findall(t))
    
    if has_generic_lottery and has_link: score += 5      # lottery scam + link is high-signal
    if has_generic_lottery and has_urgency: score += 3   # lottery + urgency
    if has_generic_phishing and has_link: score += 4     # phishing + link is classic
    if has_generic_phishing and has_pii_request: score += 4
    if has_generic_sms_spam and has_link: score += 3     # SMS spam with link
    if has_generic_sms_spam and has_urgency: score += 2  # SMS spam with urgency
    if has_generic_financial and (has_link or has_payment): score += 4

    # Legacy SMS prize-bait scams often include a callback number plus "reply/call/txt" mechanics.
    # Keep this narrowly gated to avoid boosting benign "free + call me" community chatter.
    if (
        "phone_number" in matched
        and has_legacy_sms_action
        and (
            has_legacy_sms_promo_strong
            or (has_legacy_sms_free and has_link)
        )
    ):
        score += 4

    # Shortcode-driven SMS promos (TXT/REPLY to 5-digit number + promo bait) are strong smishing signals.
    # This avoids over-triggering by requiring both an action verb and explicit promo/prize language.
    if has_sms_shortcode and has_sms_shortcode_action and has_sms_shortcode_promo:
        score += 5

    # Bank-account security phishing cluster without explicit links (common in legacy email spam corpora).
    # Require both phishing phrase and at least two account-security context terms for precision.
    if has_account_security_phish_signal and account_security_context_hits >= 2:
        score += 5

    # Lottery bait routed to a direct email address should escalate even without a link.
    # This catches manual-contact prize scams that the sidebar Gemini test already flags.
    if has_lottery and "email_address" in matched:
        score += 3

    # SAFE HARBOR: awareness / anti-scam discussion should not be treated like the scam it is describing
    if has_awareness_safe_context and not offplatform and not has_payment:
        score -= 3

    score = max(score, 0)

    seen = set()
    matched_unique =[]
    for m in matched:
        if m not in seen:
            matched_unique.append(m)
            seen.add(m)
    return {"heuristic_score": int(score), "matched_terms": matched_unique}

def should_escalate_tier1(heuristic_result: dict) -> bool:
    if not isinstance(heuristic_result, dict):
        return False
    score = int(heuristic_result.get("heuristic_score", 0) or 0)
    matched_terms = heuristic_result.get("matched_terms") or []
    return score >= SCAM_TIER1_SCORE_THRESHOLD or (SCAM_TIER1_SHORTLINK_TOKEN in matched_terms)

def classify_fraud_severity(heuristic_result: dict) -> tuple:
    """
    3-TIER IMPROVEMENT: Classify fraud signals into severity levels for precision-optimized escalation.
    
    Returns: (should_escalate, severity_level)
      - severity_level: "HIGH" (score ≥15), "MEDIUM" (score 8-14), "LOW" (score 5-7), or None (skip)
      - should_escalate: bool
    
    **HIGH_SEVERITY:** Explicit fraud patterns (bait+payment+offplatform, impersonation+PII/gift cards)
    **MEDIUM_SEVERITY:** Single strong pattern + routing, or payment + urgency
    **LOW_SEVERITY:** Suspicious keywords + metadata (links, emails, shortlinks)
    """
    if not isinstance(heuristic_result, dict):
        return False, None
    
    score = int(heuristic_result.get("heuristic_score", 0) or 0)
    matched_terms = set(heuristic_result.get("matched_terms") or [])
    
    # Shortlink is always HIGH severity (Tier 1 fast escalation)
    if "shortlink" in matched_terms:
        return True, "HIGH"
    
    # Score-based severity classification
    if score >= 15:
        severity = "HIGH"
    elif score >= 8:
        severity = "MEDIUM"
    elif score >= 5:
        severity = "LOW"
    else:
        return False, None  # Skip low-signal items
    
    return True, severity

@st.cache_data(show_spinner=False, ttl=60*60)
def _gemini_scam_classify_cached(comment_text: str) -> str:
    gemini_model, gemini_available = load_gemini_model(FAST_REVIEW_MODEL_NAME)
    if not gemini_available: return "ERROR: GEMINI_UNAVAILABLE"
    prompt = SCAM_CHECK_PROMPT_TEMPLATE.format(comment_text=comment_text)
    try:
        resp = generate_with_retry(gemini_model, prompt, generation_config=build_generation_config(response_mime_type="application/json"))
        return extract_json_from_text(resp.text or "")
    except Exception as e:
        return f"ERROR: {str(e)}"

@st.cache_data(show_spinner=False, ttl=86400) 
def _gemini_sanity_check_cached(comment_text: str) -> str:
    gemini_model, gemini_available = load_gemini_model(FAST_REVIEW_MODEL_NAME)
    if not gemini_available: return ""
    prompt = SANITY_CHECK_PROMPT_TEMPLATE.format(comment_text=comment_text)
    try:
        resp = generate_with_retry(gemini_model, prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"ERROR: {e}"

@st.cache_data(show_spinner=False, ttl=86400)
def _gemini_sanity_check_with_context_cached(comment_text: str, model_context: str = "") -> str:
    gemini_model, gemini_available = load_gemini_model(FAST_REVIEW_MODEL_NAME)
    if not gemini_available:
        return ""

    base_prompt = SANITY_CHECK_PROMPT_TEMPLATE.format(comment_text=comment_text)
    if model_context:
        base_prompt += (
            "\n\n### Additional model signals (advisory; do not blindly trust)\n"
            f"{model_context}\n"
            "Use these signals only as supporting evidence when deciding if the comment reflects a true member need."
        )
    try:
        resp = generate_with_retry(gemini_model, base_prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"ERROR: {e}"


@st.cache_data(show_spinner=False, ttl=60*60)
def _gemini_fraud_confidence_justification_cached(comment_text: str) -> str:
    """
    Enhanced Gemini prompt for fraud classification with explicit confidence explanation.

    Parses Gemini response into: {is_scam, confidence, confidence_reasoning, indicators, recommended_action}
    """
    gemini_model, gemini_available = load_gemini_model(FAST_REVIEW_MODEL_NAME)
    if not gemini_available:
        return ""

    enhanced_prompt = f"""{SCAM_CHECK_PROMPT_TEMPLATE.format(comment_text=comment_text)}

### CONFIDENCE JUSTIFICATION
Provide your confidence score (0.0-1.0) with brief reasoning:
- 0.85-1.0 (Very High): Clear, multiple fraud indicators (bait+payment+routing, or explicit impersonation)
- 0.70-0.84 (High): Strong single pattern or clear but isolated fraud signal
- 0.50-0.69 (Medium): Ambiguous; could be legitimate or fraud (military-context or peer-support?)
- <0.50 (Low): Likely legitimate; scam concerns unsubstantiated

Add to your JSON response:
{{
  "is_scam": true/false,
  "confidence": <float 0.0-1.0>,
  "confidence_reasoning": "<brief explanation of confidence level>",
  "indicators": [<list of fraud indicators>],
  "recommended_action": "<block|flag_for_review|allow>"
}}
"""

    try:
        resp = generate_with_retry(
            gemini_model,
            enhanced_prompt,
            generation_config=build_generation_config(response_mime_type="application/json")
        )
        return extract_json_from_text(resp.text or "")
    except Exception as e:
        return f"ERROR: {str(e)}"

@st.cache_data(show_spinner=False, ttl=60*60)
def _gemini_military_context_check_cached(comment_text: str) -> str:
    """
    Military-family context agent: Re-evaluates flagged items to reduce false positives
    on legitimate military-family discussions (PCS, deployment, EFMP, VA benefits, etc.).
    """
    gemini_model, gemini_available = load_gemini_model(FAST_REVIEW_MODEL_NAME)
    if not gemini_available:
        return '{"reasoning": "Gemini unavailable", "context_verdict": "unknown", "confidence": 0.0, "bsf_pattern": "N/A"}'
    
    prompt = BSF_MILITARY_CONTEXT_PROMPT_TEMPLATE.format(comment_text=comment_text)
    try:
        resp = generate_with_retry(
            gemini_model, 
            prompt, 
            generation_config=build_generation_config(response_mime_type="application/json")
        )
        return extract_json_from_text(resp.text or "")
    except Exception as e:
        return f'ERROR: {str(e)}'

@st.cache_data(show_spinner=False, ttl=86400)
def _gemini_support_eval_cached(comment_text: str) -> str:
    gemini_model, gemini_available = load_gemini_model(TEACHER_MODEL_NAME)
    if not gemini_available:
        gemini_model, gemini_available = load_gemini_model(FAST_REVIEW_MODEL_NAME)
    if not gemini_available:
        return ""
    prompt = SANITY_CHECK_PROMPT_TEMPLATE.format(comment_text=comment_text)
    try:
        resp = generate_with_retry(gemini_model, prompt)
        return (resp.text or "").strip()
    except Exception as e:
        return f"ERROR: {e}"

def categorize_scam_severity(scam_flags):
    """
    Stratify scam flags by confidence-based severity for risk-prioritized review.
    Reads A/B experiment thresholds from st.session_state when available.

    Severity Levels:
    - CRITICAL: is_scam=true, confidence >= block_threshold, high-risk payment + off-platform contact
    - HIGH: is_scam=true, confidence 0.75–block_threshold, predatory indicators present
    - MEDIUM: is_scam=true, confidence medium_threshold–0.74, uncertain but suspicious
    - LOW: catch-all
    """
    block_threshold  = st.session_state.get("ab_block_threshold",  0.85)
    medium_threshold = st.session_state.get("ab_medium_threshold", 0.60)
    severity_buckets = {"critical": [], "high": [], "medium": [], "low": []}
    
    for flag in scam_flags:
        conf = flag.get("confidence") or 0.0
        is_scam = flag.get("is_scam", False)
        action = flag.get("recommended_action", "")
        indicators = flag.get("indicators", [])
        
        # Identify high-risk patterns
        has_payment_terms = any(t in indicators for t in HIGH_RISK_PAYMENT_TERMS)
        has_offplatform = any(t in indicators for t in OFFPLATFORM_CONTACT_TERMS)
        has_bait = any(t in indicators for t in ("giving away", "giveaway", "won", "winner", "claim your"))
        
        high_floor = max(medium_threshold, 0.75)   # HIGH always above MEDIUM floor
        if is_scam and conf >= block_threshold and (has_payment_terms or (has_offplatform and has_bait)):
            severity_buckets["critical"].append({**flag, "severity": "CRITICAL", "review_priority": 1})
        elif is_scam and conf >= high_floor:
            severity_buckets["high"].append({**flag, "severity": "HIGH", "review_priority": 2})
        elif is_scam and conf >= medium_threshold:
            severity_buckets["medium"].append({**flag, "severity": "MEDIUM", "review_priority": 3})
        else:
            severity_buckets["low"].append({**flag, "severity": "LOW", "review_priority": 4})
    
    return severity_buckets

def apply_tier3_context_gates(candidate: dict, df: pd.DataFrame = None) -> dict:
    """
    TIER 3: Context-aware refinement to reduce false positives on legitimate posts.
    
    Applies BSF-specific safety gates:
    1. **Peer-support pattern:** "I help with X" or mentoring language -> safe (not predatory)
    2. **Business legitimacy:** concrete service + contact info + reasonable post -> safe
    3. **Scam awareness:** "is this a scam?", "warning" -> safe (discussing scams, not perpetrating)
    4. **Account age:** metadata check (if df provided) for new account red flags
    
    Returns: dict with `tier3_verdict` (str) and `tier3_reason` (str)
    """
    comment = str(candidate.get("comment", "")).lower() if pd.notna(candidate.get("comment")) else ""
    commenter_name = str(candidate.get("commenter_name", "")).lower() if pd.notna(candidate.get("commenter_name")) else ""
    
    result = {
        "tier3_verdict": "proceed",  # Default: continue through normal pipeline
        "tier3_reason": ""
    }
    
    # GATE 1: Peer-support / mentoring pattern
    peer_support_patterns = [
        r"\bi (?:can )?help with", r"\bi mentor", r"\bi assist", r"\bi offer guidance",
        r"i (?:volunteer|volunteer to help|provide support)", r"\bi have experience with",
        r"\bi (?:trained|specialize|focus) in"
    ]
    if any(re.search(pat, comment) for pat in peer_support_patterns):
        # Check for predatory signals (payment, off-platform routing)
        has_predatory = any(term in comment for term in ["gift card", "send me dm", "dm me", "whatsapp", "telegram"])
        if not has_predatory:
            result["tier3_verdict"] = "safe_peer_support"
            result["tier3_reason"] = "Peer-support pattern detected; no predatory signals"
            return result
    
    # GATE 2: Business legitimacy (service offering)
    business_signals = [
        ("service offering", r"(i (?:offer|provide|sell|offer to)\s+(?:sell|provide|offer|make|create|design|build)\b)"),
        ("concrete description", r"((?:handmade|custom|vintage|professional|licensed|insured)\b)"),
        ("pricing transparency", r"(\$\d+|cost.*\d+|price.*\d+|fee.*\d+)"),
        ("contact methods", r"((?:email|phone|website|facebook|etsy|shopify)\b)"),
    ]
    business_score = 0
    business_reasons = []
    for signal_name, pattern in business_signals:
        if re.search(pattern, comment):
            business_score += 1
            business_reasons.append(signal_name)
    
    # If 3+ business legitimacy signals AND no predatory routing, likely safe
    has_offplatform = any(term in comment for term in ["send me dm", "dm me", "whatsapp", "telegram", "signal"])
    if business_score >= 3 and not has_offplatform:
        result["tier3_verdict"] = "safe_business"
        result["tier3_reason"] = f"Business legitimacy signals: {', '.join(business_reasons)}"
        return result
    
    # GATE 3: Scam awareness / warning discussions
    awareness_patterns = [
        r"\b(?:is this a scam\?|warning|don't click|do not click|beware|scam alert|be careful|watch out|this is a scam|fake)\b",
        r"\b(?:is it legit|how to spot|red flags|seems suspicious|sounds like scam)\b"
    ]
    if any(re.search(pat, comment) for pat in awareness_patterns):
        # If user is discussing scams without payment/routing, it's safe-harbor
        has_predatory = any(term in comment for term in ["gift card", "send me dm", "dm me", "upfront payment"])
        if not has_predatory:
            result["tier3_verdict"] = "safe_awareness"
            result["tier3_reason"] = "User is warning others about scams; not perpetrating"
            return result
    
    # GATE 4: Metadata safety (account age, post history)
    # Note: requires df; if not provided, skip
    idx = candidate.get("index")
    if df is not None and idx in df.index:
        try:
            row = df.loc[idx]
            # Check for user pattern indicators (if available)
            user_post_count = row.get("user_post_count", 1)
            account_days_old = row.get("account_days_old", 365)  # Default: assume 1 year
            
            # New account (< 7 days) + first post = higher risk
            if account_days_old < 7 and user_post_count == 1:
                result["tier3_verdict"] = "new_account_risk"
                result["tier3_reason"] = f"New account ({account_days_old}d) with first post; elevated risk"
                # Don't override; just mark for extra caution
        except Exception:
            pass  # Metadata not available; continue
    
    # Default: no context override; proceed through normal pipeline
    return result


def detect_scam_concerns(
    df: pd.DataFrame,
    silent: bool = False,
    progress_bar=None,
    status_container=None,
    progress_ui=None,
    include_trusted_users: bool = False,
):
    if df is None or df.empty: return[]
    text_col = "content" if "content" in df.columns else None
    if not text_col: return[]

    # A/B experiment thresholds now affect core escalation policy (not just display severity)
    block_threshold = float(st.session_state.get("ab_block_threshold", 0.85))
    medium_threshold = float(st.session_state.get("ab_medium_threshold", 0.60))
    if medium_threshold >= block_threshold:
        medium_threshold = max(0.50, round(block_threshold - 0.01, 2))

    total_comments = len(df)
    initial_candidates =[]
    if not silent and progress_ui:
        progress_ui.mark_step_started(3, f'Scanning {total_comments} interactions with deterministic regex and metadata heuristics...')

    for i, row in df.iterrows():
        if not silent and progress_ui:
            progress_ui.update_step(3, i + 1, total_comments, f'Heuristic scan {i+1}/{total_comments}')
        elif not silent and progress_bar and status_container:
            status_container.markdown(f"⏳ **Step 3/4: Scam Check (Tier 1)** Scanning {i+1}/{total_comments}")
            progress_bar.progress((i + 1) / total_comments)

        commenter_name = row.get("commenter_name", "")
        commenter_email = row.get("commenter_email", "")

        if (not include_trusted_users) and is_staff_or_admin_comment(commenter_name, commenter_email):
            continue

        comment = str(row.get(text_col, "") or "")
        h = scam_heuristic_scan(comment)

        # TIER 1: Use severity classification (HIGH/MEDIUM/LOW) for precision-focused escalation
        # HIGH (≥15): 100% escalate | MEDIUM (8-14): 60% escalate | LOW (5-7): 40% escalate
        should_escalate, severity_level = classify_fraud_severity(h)
        if should_escalate:
            initial_candidates.append({
                "index": int(i) if isinstance(i, (int, np.integer)) else i,
                "comment": comment,
                "scan_week": normalize_identity_value(row.get("scam_scan_week", "Selected week"), fallback="Selected week"),
                "community_name": normalize_identity_value(row.get("community_name", "N/A"), fallback="N/A"),
                "commenter_name": normalize_identity_value(row.get("commenter_name", "N/A"), fallback="Unknown"),
                "commenter_email": normalize_identity_value(row.get("commenter_email", "N/A"), fallback="N/A"),
                "detected_contact_email": extract_detected_contact_email(comment),
                "heuristic_score": h["heuristic_score"],
                "matched_terms": h["matched_terms"],
                "severity_level": severity_level,
            })

    if not initial_candidates:
        if not silent and progress_ui:
            progress_ui.complete_step(3, 'Heuristic scan complete: no suspicious items found.')
            progress_ui.skip_step(4, 'No Gemini verification needed for scam detection.')
        return[]

    flags =[]
    comments_to_process = [item["comment"] for item in initial_candidates]

    def _should_escalate_without_llm(item):
        terms = set(item.get("matched_terms") or [])
        has_payment = any(term in terms for term in HIGH_RISK_PAYMENT_TERMS)
        has_offplatform = any(term in terms for term in OFFPLATFORM_CONTACT_TERMS)
        has_bait = any(term in terms for term in ("giving away", "giveaway", "won", "winner", "claim your"))
        return bool(item.get("heuristic_score", 0) >= 8 and (has_payment and has_offplatform or has_bait and has_offplatform))

    llm_unavailable = False
    llm_error_reason = ""
    
    gemini_probe = _gemini_scam_classify_cached("health probe")
    if isinstance(gemini_probe, str) and gemini_probe.startswith("ERROR:"):
        llm_unavailable = True
        llm_error_reason = gemini_probe
        raw_results = {}
    else:
        if not silent and progress_ui:
            progress_ui.complete_step(3, f'Heuristic scan complete: {len(initial_candidates)} item(s) sent to Gemini review.')
            progress_ui.mark_step_started(4, f'Gemini Trust & Safety reviewer analyzing {len(comments_to_process)} suspicious item(s) for predatory off-platform routing...')
            raw_results = process_in_parallel(
                _gemini_scam_classify_cached,
                comments_to_process,
                progress_callback=lambda completed, total: progress_ui.update_step(4, completed, total, f'Gemini reviewed {completed}/{total} suspicious item(s)')
            )
            progress_ui.complete_step(4, f'Gemini scam review complete for {len(comments_to_process)} item(s).')
        elif not silent and progress_bar and status_container:
            progress_bar.progress(0.0)
            raw_results = process_in_parallel(
                _gemini_scam_classify_cached,
                comments_to_process,
                progress_bar,
                status_container,
                "⏳ **Step 4/4: Scam Check (Tier 2)** Gemini verified {completed}/{total} suspicious items"
            )
        else:
            raw_results = process_in_parallel(_gemini_scam_classify_cached, comments_to_process)

    if not silent:
        st.session_state['scam_review_mode'] = "Tier 1 + Tier 2 (Gemini)" if not llm_unavailable else "Tier 1 fallback (Gemini unavailable)"
        st.session_state['scam_review_error'] = llm_error_reason if llm_unavailable else ""
        
    for j, item in enumerate(initial_candidates):
        out = {**item, "is_scam": None, "confidence": None, "recommended_action": "flag_for_review", "reasons":[], "indicators":[], "llm_status": "OK"}
        raw_text = raw_results.get(j, "ERROR: Timeout") if not llm_unavailable else llm_error_reason
        
        if not raw_text.startswith("ERROR:"):
            try:
                r = json.loads(raw_text)
                out["is_scam"] = bool(r.get("is_scam")) if r.get("is_scam") is not None else None
                try: out["confidence"] = float(r.get("confidence")) if r.get("confidence") is not None else None
                except: out["confidence"] = None
                out["recommended_action"] = r.get("recommended_action") or out["recommended_action"]
                out["reasons"] = [r.get("reasoning", "")] + (r.get("reasons") or [])
                out["indicators"] = r.get("indicators") or[]
            except:
                out["llm_status"] = "JSON_PARSE_ERROR"
        else:
            out["llm_status"] = raw_text
            severity_level = str(item.get("severity_level") or "").upper()
            fallback_positive = severity_level in {"HIGH", "MEDIUM"} or _should_escalate_without_llm(item)
            out["is_scam"] = True if fallback_positive else None
            if out["is_scam"]:
                if severity_level == "HIGH":
                    out["confidence"] = 0.9
                    out["recommended_action"] = "block"
                elif severity_level == "MEDIUM":
                    out["confidence"] = 0.75
                    out["recommended_action"] = "flag_for_review"
                else:
                    out["confidence"] = 0.55
                    out["recommended_action"] = "audit"
            else:
                out["confidence"] = 0.2
                out["recommended_action"] = "audit"
            out["indicators"] = item["matched_terms"]
            out["reasons"] =["Gemini unavailable; conservative heuristic fallback applied"]

        # Reliability fallback: when Gemini is unavailable, do not silently drop Tier-1 positives.
        # Escalate HIGH/MEDIUM severity candidates; send LOW severity to audit queue.
        if out["llm_status"] != "OK":
            severity_level = str(item.get("severity_level") or "").upper()
            if out.get("is_scam") is True and severity_level in {"HIGH", "MEDIUM"}:
                flags.append(out)
            else:
                out["audit_reason"] = "Gemini unavailable: low-confidence or low-severity fallback"
                if "audit_queue" not in st.session_state:
                    st.session_state["audit_queue"] = []
                st.session_state["audit_queue"].append(out)
            continue

        # TIER 3 (NEW): Context-aware gates reduce false positives on legitimate posts
        # (peer-support, business offerings, scam-awareness discussions, account metadata)
        tier3_result = apply_tier3_context_gates(out, df)
        if tier3_result.get("tier3_verdict"):
            out["tier3_verdict"] = tier3_result["tier3_verdict"]
            out["tier3_reason"] = tier3_result.get("tier3_reason", "")
            # Demote verdict if context gates pass: BLOCK->REVIEW, REVIEW->AUDIT
            if out["recommended_action"] == "block" and tier3_result["tier3_verdict"].startswith("safe_"):
                out["recommended_action"] = "flag_for_review"
                out["reasons"].append(f"Tier 3 context override: {tier3_result['tier3_reason']}")
            elif out["recommended_action"] == "flag_for_review" and tier3_result["tier3_verdict"] == "new_account_risk":
                out["audit_reason"] = f"Tier 3 metadata flag: {tier3_result['tier3_reason']}"
                out["reasons"].append(f"Tier 3 metadata flag: {tier3_result['tier3_reason']}")

        # TIER 2 FILTERING: Respect Gemini verdicts and A/B block threshold.
        # Only flag if: (1) Confident scam at/above block threshold OR (2) explicit block.
        # Do NOT flag just because Tier 1 matched. "flag_for_review" items go to audit queue.
        # None confidence + no explicit block → audit queue, not auto-escalate (improves precision).
        is_confident_scam = out["is_scam"] is True and out["confidence"] is not None and out["confidence"] >= block_threshold
        is_explicit_block = out["recommended_action"] == "block"
        
        # TIER 2.5 (NEW): Military-context filtering for medium-confidence flags
        # If confidence is in [medium_threshold, block_threshold) and is_scam=true,
        # apply military-context check to reduce false positives on legitimate military-family posts.
        requires_military_check = out["is_scam"] is True and medium_threshold <= (out["confidence"] or 0.0) < block_threshold
        military_context_pass = False
        
        if requires_military_check and out["llm_status"] == "OK":
            try:
                military_raw = _gemini_military_context_check_cached(item["comment"])
                if not military_raw.startswith("ERROR:"):
                    military_verdict = json.loads(military_raw)
                    military_context_pass = military_verdict.get("context_verdict") == "context_pass"
                    out["military_context_verdict"] = military_verdict.get("context_verdict")
                    out["military_context_pattern"] = military_verdict.get("bsf_pattern", "unknown")
                    if military_context_pass:
                        out["reasons"].insert(0, f"Military context agent: {military_verdict.get('reasoning', 'Legitimate military-family pattern')}")
            except Exception as e:
                out["military_context_error"] = str(e)
        
        if is_confident_scam or is_explicit_block:
            # Only suppress if military context says it's legitimate
            if not (requires_military_check and military_context_pass):
                flags.append(out)
            else:
                # Military context suppressed this medium-confidence flag
                out["audit_reason"] = f"Military context override: {out.get('military_context_pattern', 'legitimate pattern')}"
                if "audit_queue" not in st.session_state:
                    st.session_state["audit_queue"] = []
                st.session_state["audit_queue"].append(out)
        elif out["recommended_action"] == "flag_for_review" and out["llm_status"] == "OK":
            # Track unclear cases for audit/model improvement (separate from scam_flags)
            out["audit_reason"] = "Tier 1 matched but Gemini unclear; low confidence for escalation"
            if "audit_queue" not in st.session_state:
                st.session_state["audit_queue"] = []
            st.session_state["audit_queue"].append(out)

    if llm_unavailable and not silent:
        st.warning(
            "Gemini scam verification is unavailable (likely API key issue). "
            "Scam output is running in conservative Tier-1 fallback mode to reduce false positives."
        )

    return flags

def detect_false_negatives_sampling(df: pd.DataFrame, flagged_indices: set, sample_size=50, silent=False):
    """
    Safety audit: Sample posts that PASSED (not flagged) and do a spot-check for missed scams.
    This measures FALSE NEGATIVE RATE (scams that got through) for operational safety tracking.
    
    Returns:
      - false_negatives: list of posts that passed but are likely scams
      - sample_stats: metrics on pass-rate, detection confidence, etc.
    """
    if df is None or df.empty:
        return [], {"sample_size": 0, "false_negative_rate": 0.0, "error": "No data"}
    
    text_col = "content" if "content" in df.columns else None
    if not text_col:
        return [], {"sample_size": 0, "false_negative_rate": 0.0, "error": "No content column"}
    
    # Identify posts that PASSED (not flagged)
    passed_indices = [i for i in range(len(df)) if i not in flagged_indices]
    if not passed_indices:
        return [], {"sample_size": 0, "false_negative_rate": 0.0, "note": "All posts flagged"}
    
    # Random sample
    sample_indices = random.sample(passed_indices, min(sample_size, len(passed_indices)))
    
    false_negatives = []
    gemini_available = True
    
    for idx in sample_indices:
        comment = str(df.iloc[idx].get(text_col, "") or "")
        if not comment.strip():
            continue
        
        # Run Gemini scam check on passed item
        gemini_result = _gemini_scam_classify_cached(comment)
        
        if gemini_result.startswith("ERROR:"):
            gemini_available = False
            continue
        
        try:
            result_json = json.loads(gemini_result)
            is_scam = result_json.get("is_scam", False)
            confidence = result_json.get("confidence", 0.0)
            
            # Flag if confidence >= 0.80 for missed scams
            if is_scam and confidence >= 0.80:
                false_negatives.append({
                    "index": idx,
                    "comment": comment,
                    "commenter_name": df.iloc[idx].get("commenter_name", "N/A"),
                    "community_name": df.iloc[idx].get("community_name", "N/A"),
                    "confidence": confidence,
                    "reasoning": result_json.get("reasoning", ""),
                    "indicators": result_json.get("indicators", [])
                })
        except Exception:
            continue
    
    false_negative_rate = len(false_negatives) / max(len(sample_indices), 1)
    
    stats = {
        "sample_size": len(sample_indices),
        "passed_total": len(passed_indices),
        "false_negatives_detected": len(false_negatives),
        "false_negative_rate": round(false_negative_rate * 100, 2),
        "gemini_available": gemini_available
    }
    
    if not silent:
        st.session_state["false_negative_stats"] = stats
    
    return false_negatives, stats

@st.cache_data(show_spinner=False)
def load_scam_benchmark_corpus() -> list[dict]:
    if not SCAM_BENCHMARK_FILE.exists():
        return []
    try:
        data = json.loads(SCAM_BENCHMARK_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return data

def _compute_benchmark_metrics(items: list[dict], pred_field: str) -> dict:
    total = len(items)
    actual_positive = sum(1 for item in items if item.get("expected_scam"))
    actual_negative = total - actual_positive
    tp = sum(1 for item in items if item.get("expected_scam") and item.get(pred_field))
    fp = sum(1 for item in items if not item.get("expected_scam") and item.get(pred_field))
    tn = sum(1 for item in items if not item.get("expected_scam") and not item.get(pred_field))
    fn = sum(1 for item in items if item.get("expected_scam") and not item.get(pred_field))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / actual_positive if actual_positive else 0.0
    specificity = tn / actual_negative if actual_negative else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    false_positive_rate = fp / actual_negative if actual_negative else 0.0

    return {
        "total": total,
        "scam_examples": actual_positive,
        "legit_examples": actual_negative,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "accuracy": accuracy,
        "false_positive_rate": false_positive_rate,
        "precision_pct": round(precision * 100, 1),
        "recall_pct": round(recall * 100, 1),
        "specificity_pct": round(specificity * 100, 1),
        "accuracy_pct": round(accuracy * 100, 1),
        "false_positive_rate_pct": round(false_positive_rate * 100, 1),
    }

def _summarize_benchmark_categories(items: list[dict]) -> pd.DataFrame:
    rows = []
    for category in sorted({item.get("category", "unknown") for item in items}):
        bucket = [item for item in items if item.get("category") == category]
        scam_total = sum(1 for item in bucket if item.get("expected_scam"))
        legit_total = len(bucket) - scam_total
        tier1_hits = sum(1 for item in bucket if item.get("tier1_flag") and item.get("expected_scam"))
        tier1_fp = sum(1 for item in bucket if item.get("tier1_flag") and not item.get("expected_scam"))
        pipeline_hits = sum(1 for item in bucket if item.get("pipeline_flag") and item.get("expected_scam"))
        pipeline_fp = sum(1 for item in bucket if item.get("pipeline_flag") and not item.get("expected_scam"))
        rows.append({
            "category": category,
            "examples": len(bucket),
            "scams": scam_total,
            "legit": legit_total,
            "tier1_hits": tier1_hits,
            "tier1_fp": tier1_fp,
            "pipeline_hits": pipeline_hits,
            "pipeline_fp": pipeline_fp,
        })
    return pd.DataFrame(rows)

def evaluate_scam_benchmark(max_items: int | None = None, run_full_pipeline: bool = False) -> dict:
    corpus = load_scam_benchmark_corpus()
    if max_items:
        corpus = corpus[:max_items]

    benchmark_items = []
    for idx, entry in enumerate(corpus):
        text_value = str(entry.get("text", "") or "")
        heuristics = scam_heuristic_scan(text_value)
        tier1_flag = should_escalate_tier1(heuristics)
        benchmark_items.append({
            "row_index": idx,
            "id": entry.get("id", f"item_{idx}"),
            "label": entry.get("label", "legit"),
            "expected_scam": entry.get("label") == "scam",
            "category": entry.get("category", "unknown"),
            "source_name": entry.get("source_name", "unknown"),
            "source_url": entry.get("source_url", ""),
            "text": text_value,
            "heuristic_score": heuristics.get("heuristic_score", 0),
            "matched_terms": ", ".join(heuristics.get("matched_terms") or []),
            "tier1_flag": bool(tier1_flag),
            "pipeline_flag": False,
        })

    audit_queue_backup = list(st.session_state.get("audit_queue", [])) if hasattr(st, "session_state") else []
    audit_queue_count = 0
    if run_full_pipeline and benchmark_items:
        benchmark_df = pd.DataFrame([
            {
                "content": item["text"],
                "commenter_name": f"benchmark-user-{item['row_index']}",
                "commenter_email": f"benchmark{item['row_index']}@example.com",
                "community_name": item["category"],
            }
            for item in benchmark_items
        ])
        try:
            st.session_state["audit_queue"] = []
            pipeline_flags = detect_scam_concerns(benchmark_df, silent=True)
            flagged_indices = {int(flag.get("index")) for flag in pipeline_flags if isinstance(flag.get("index"), (int, np.integer))}
            audit_queue_count = len(st.session_state.get("audit_queue", []))
        finally:
            st.session_state["audit_queue"] = audit_queue_backup

        for item in benchmark_items:
            item["pipeline_flag"] = item["row_index"] in flagged_indices

    tier1_metrics = _compute_benchmark_metrics(benchmark_items, "tier1_flag")
    pipeline_metrics = _compute_benchmark_metrics(benchmark_items, "pipeline_flag") if run_full_pipeline else None
    category_summary = _summarize_benchmark_categories(benchmark_items)
    result_df = pd.DataFrame(benchmark_items)

    return {
        "benchmark_items": benchmark_items,
        "tier1_metrics": tier1_metrics,
        "pipeline_metrics": pipeline_metrics,
        "category_summary": category_summary,
        "audit_queue_count": audit_queue_count,
        "run_full_pipeline": run_full_pipeline,
        "results_csv": result_df.to_csv(index=False).encode("utf-8"),
    }

def detect_severe_concerns(df_em, df, silent=False, progress_bar=None, status_container=None, progress_ui=None):
    initially_flagged = []

    for i, row in df_em.iterrows():
        original_row = df.iloc[i]
        if is_staff_or_admin_comment(original_row.get('commenter_name', ''), original_row.get('commenter_email', '')):
            continue
        comment_text = original_row.get('content', '')
        candidate_details = compute_severe_candidate_details(row, comment_text)

        if candidate_details['candidate']:
            initially_flagged.append({
                'index': i + 1,
                'emotion': candidate_details['dominant_emotion'].title() if candidate_details['dominant_emotion'] else 'Mixed',
                'score': round(candidate_details['dominant_score'] * 100, 1),
                'total_negative': round(candidate_details['severe_sum'] * 100, 1),
                'avg_negative_intensity': round(candidate_details.get('severe_avg', 0.0) * 100, 1),
                'negative_component_count': candidate_details.get('severe_component_count', 0),
                'negative_components': [emotion.title() for emotion in candidate_details.get('severe_components', [])],
                'reason_flags': candidate_details['reason_flags'],
                'rare_hints': candidate_details['rare_hints'],
                'support_need_hits': candidate_details['support_need_hits'],
                'comment': comment_text,
                'community_name': original_row.get('community_name', 'N/A'),
                'commenter_name': original_row.get('commenter_name', 'N/A'),
                'commenter_email': original_row.get('commenter_email', 'N/A')
            })

    if not silent:
        try:
            thr = get_label_thresholds()
            expected_idxs = [8, 38, 64, 91, 94, 106, 107, 137, 145, 156, 166, 168, 186]
            key_emos = ["grief", "sadness", "fear", "disappointment", "nervousness", "remorse", "embarrassment", "neutral"]
            scores_for_expected = {}
            for idx in expected_idxs:
                if 0 <= idx < len(df_em):
                    r = df_em.iloc[idx]
                    scores_for_expected[idx] = {e: round(float(r.get(e, 0.0)), 3) for e in key_emos}
            st.session_state['last_severe_debug'] = {
                "df_rows": int(len(df)),
                "df_em_rows": int(len(df_em)),
                "flagged_count": len(initially_flagged),
                "flagged_indices": [c["index"] for c in initially_flagged],
                "flagged_reasons": [c["reason_flags"] for c in initially_flagged],
                "thresholds_key": {e: thr.get(e) for e in key_emos},
                "expected_idx_scores": scores_for_expected,
                "expected_idx_content": {idx: str(df.iloc[idx].get("content", ""))[:120] for idx in expected_idxs if 0 <= idx < len(df)},
            }
        except Exception:
            pass

    if not initially_flagged:
        if not silent and progress_ui:
            progress_ui.skip_step(2, 'No severe interactions required Gemini review.')
        return [], []

    gemini_verified_concerns = []
    def _build_model_context(concern):
        return (
            f"dominant_emotion={concern.get('emotion', 'N/A')}; "
            f"dominant_score={concern.get('score', 0)}%; "
            f"cumulative_negative_load={concern.get('total_negative', 0)} pts; "
            f"avg_negative_intensity={concern.get('avg_negative_intensity', 0)}%; "
            f"reason_flags={', '.join(concern.get('reason_flags', [])) or 'none'}; "
            f"rare_hints={', '.join(concern.get('rare_hints', [])) or 'none'}; "
            f"support_need_hits={len(concern.get('support_need_hits', []))}"
        )

    comments_to_process = [
        (concern['comment'], _build_model_context(concern))
        for concern in initially_flagged
    ]

    def _sanity_worker(payload):
        comment_text, context_text = payload
        return _gemini_sanity_check_with_context_cached(comment_text, context_text)

    if not silent and progress_ui:
        progress_ui.mark_step_started(2, f'Gemini precision filter reviewing {len(comments_to_process)} severe candidate(s) using Chain-of-Thought reasoning...')
        raw_results = process_in_parallel(
            _sanity_worker,
            comments_to_process,
            progress_callback=lambda completed, total: progress_ui.update_step(2, completed, total, f'Gemini verified {completed}/{total} severe interaction(s)')
        )
        progress_ui.complete_step(2, f'AI sanity check complete for {len(comments_to_process)} interaction(s).')
    elif not silent and progress_bar and status_container:
        progress_bar.progress(0.0)
        raw_results = process_in_parallel(
            _sanity_worker,
            comments_to_process,
            progress_bar,
            status_container,
            "⏳ **Step 2/4: AI Sanity Check** Gemini verified {completed}/{total} severe items"
        )
    else:
        raw_results = process_in_parallel(_sanity_worker, comments_to_process)

    api_error_count = 0
    first_api_error = ""

    for i, concern in enumerate(initially_flagged):
        raw_text = raw_results.get(i, "ERROR: Timeout")

        if isinstance(raw_text, str) and raw_text.startswith("ERROR:"):
            api_error_count += 1
            if not first_api_error:
                first_api_error = raw_text
            if "sanity_log" in st.session_state and not silent:
                st.session_state.sanity_log.append({
                    "timestamp": datetime.now().strftime('%H:%M:%S'),
                    "comment_index": concern.get("index"),
                    "comment_preview": concern.get("comment", "")[:120] + "...",
                    "full_comment": concern.get("comment", ""),
                    "flagged_for": concern.get("emotion", "N/A"),
                    "raw_response": "API_ERROR",
                    "interpreted_label": False,
                    "status": raw_text,
                })
            continue

        verdict = parse_gemini_boolean_verdict(raw_text)
        ambiguous = verdict is None

        recall_override = (
            bool(concern.get('support_need_hits'))
            or concern.get('total_negative', 0) >= 55
            or any(flag in {'support_need_signal', 'support_need_with_grief_hint', 'severe_sum_high'} for flag in concern.get('reason_flags', []))
        )

        if verdict is None:
            is_concerning = bool(recall_override)
        else:
            is_concerning = bool(verdict)
            if (verdict is False) and recall_override and concern.get('total_negative', 0) >= 70:
                is_concerning = True

        # Tag how this concern reached "True" so the UI can show only Gemini-verified True items.
        if verdict is True:
            concern['verdict_source'] = 'gemini_true'
        elif verdict is None and is_concerning:
            concern['verdict_source'] = 'recall_override_ambiguous'
        elif verdict is False and is_concerning:
            concern['verdict_source'] = 'recall_override_false'
        else:
            concern['verdict_source'] = 'gemini_false'

        if "sanity_log" in st.session_state and not silent:
            st.session_state.sanity_log.append({
                "timestamp": datetime.now().strftime('%H:%M:%S'),
                "comment_index": concern.get("index"),
                "comment_preview": concern.get("comment", "")[:120] + "...",
                "full_comment": concern.get("comment", ""),
                "flagged_for": concern.get("emotion", "N/A"),
                "raw_response": raw_text,
                "interpreted_label": is_concerning,
                "status": "Recall Override" if (ambiguous and is_concerning) else ("Ambiguous" if ambiguous else "Verified"),
            })

        if is_concerning:
            gemini_verified_concerns.append(concern)

    if api_error_count and not silent:
        st.warning(
            f"Gemini severe verification failed for {api_error_count} candidate(s). "
            f"Example error: {first_api_error}"
        )

    gemini_verified_concerns.sort(key=lambda x: x['total_negative'], reverse=True)
    return gemini_verified_concerns, initially_flagged


def count_severe_candidate_comments(df_em, df):
    if df_em is None or df is None or df_em.empty or df.empty:
        return 0

    count = 0
    for i, row in df_em.iterrows():
        text_value = ''
        if i < len(df):
            text_value = df.iloc[i].get('content', '')
        details = compute_severe_candidate_details(row, text_value)
        if details['candidate']:
            count += 1
    return count


def count_scam_tier1_candidates(df, include_trusted_users: bool = False):
    if df is None or df.empty:
        return 0

    total = 0
    text_col = "content" if "content" in df.columns else None
    if not text_col:
        return total

    for _, row in df.iterrows():
        commenter_name = row.get("commenter_name", "")
        commenter_email = row.get("commenter_email", "")

        if (not include_trusted_users) and is_staff_or_admin_comment(commenter_name, commenter_email):
            continue

        comment = str(row.get(text_col, "") or "")
        h = scam_heuristic_scan(comment)
        should_escalate, _ = classify_fraud_severity(h)
        if should_escalate:
            total += 1

    return total

def get_mild_moderate_comments(df_em, df, top_n=10):
    mild_moderate_comments = []
    for i, row in df_em.iterrows():
        original_row = df.iloc[i]
        details = compute_mild_moderate_details(row, original_row.get('content', ''))
        if details['candidate']:
            mild_moderate_comments.append({
                'index': i + 1,
                'emotion': details['dominant_emotion'].title(),
                'score': round(details['dominant_score'] * 100, 1),
                'total_negative': round(details['negative_sum'] * 100, 1),
                'avg_negative_intensity': round(details.get('negative_avg', 0.0) * 100, 1),
                'negative_component_count': details.get('negative_component_count', 0),
                'negative_components': [emotion.title() for emotion in details.get('negative_components', [])],
                'comment': original_row.get('content', ''),
                'community_name': original_row.get('community_name', 'N/A'),
                'commenter_name': original_row.get('commenter_name', 'N/A'),
                'commenter_email': original_row.get('commenter_email', 'N/A')
            })
    mild_moderate_comments.sort(key=lambda x: x['total_negative'], reverse=True)
    return mild_moderate_comments[:top_n]

def send_email_alert(flagged_comments, week, year):
    try:
        sender_email = st.secrets.email.sender_email
        sender_password = st.secrets.email.sender_password
        recipient_email = st.secrets.email.recipient_email
        smtp_server = st.secrets.email.smtp_server
        smtp_port = st.secrets.email.smtp_port
    except Exception as e:
        st.error(f"Email secrets not configured correctly in secrets.toml. Error: {e}")
        return

    subject = f"🚨 Alert: {len(flagged_comments)} Severe Interactions Detected (Week {week}, {year})"
    html_body = f"<html><head></head><body><h2 style='color: #d32d27;'>BSF Community Wellness Alert</h2><p>The automated system has detected {len(flagged_comments)} severe interactions for <strong>Week {week}, {year}</strong> that may require review. These have been verified by a secondary AI model.</p><hr>"
    for concern in flagged_comments:
        _name  = html.escape(str(concern.get('commenter_name', 'N/A')))
        _email = html.escape(str(concern.get('commenter_email', 'N/A')))
        _comm  = html.escape(str(concern.get('community_name', 'N/A')))
        _emo   = html.escape(str(concern.get('emotion', 'N/A')))
        _score = html.escape(str(concern.get('score', 'N/A')))
        _neg   = html.escape(str(concern.get('total_negative', 'N/A')))
        _text  = html.escape(str(concern.get('comment', '')))
        html_body += (f"<div style='border: 1px solid #ddd; padding: 15px; margin-bottom: 20px; border-radius: 8px; font-family: sans-serif;'>"
                      f"<p><strong>User:</strong> {_name} ({_email})</p>"
                      f"<p><strong>Community:</strong> {_comm}</p>"
                      f"<p><strong>Dominant Emotion:</strong> {_emo} ({_score}%) | <strong>Cumulative Negative Load (sum points):</strong> {_neg}%</p>"
                      f"<h4 style='margin-bottom: 5px;'>Text:</h4>"
                      f"<blockquote style='border-left: 3px solid #ccc; padding-left: 15px; margin: 0 0 0 5px; color: #333;'>{_text}</blockquote></div>")
    html_body += "</body></html>"

    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = recipient_email
    message["Subject"] = subject
    message.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, message.as_string())
        st.success(f":material/check_circle: Alert email successfully sent to {recipient_email}.")
    except Exception as e:
        st.error(f":material/cancel: Failed to send email: {e}")


def send_slack_alert(critical_flags: list, week: int, year: int) -> bool:
    """
    Post a Slack alert for CRITICAL-severity scam flags using an incoming webhook.
    Requires [slack] webhook_url in secrets.toml.  Fails silently if not configured.
    Returns True on success, False otherwise.
    """
    try:
        webhook_url = st.secrets.slack.webhook_url
    except Exception:
        return False  # Slack not configured — silently skip

    if not critical_flags:
        return False

    try:
        import urllib.request
        lines = []
        for f in critical_flags[:5]:   # cap at 5 to keep message scannable
            user   = f.get("commenter_name", "Unknown")
            comm   = f.get("community_name", "N/A")
            conf   = f.get("confidence")
            conf_s = f"{conf:.2f}" if isinstance(conf, float) else "N/A"
            indicators = ", ".join((f.get("indicators") or [])[:4]) or "N/A"
            lines.append(f"• *{user}* ({comm})  conf={conf_s}  indicators: _{indicators}_")

    
        overflow = len(critical_flags) - 5
        suffix = f"\n_…and {overflow} more critical flag(s)._" if overflow > 0 else ""

        payload = json.dumps({
            "text": (
                f":rotating_light: *BSF Scam Alert — Week {week}, {year}*\n"
                f"{len(critical_flags)} CRITICAL-severity flag(s) require immediate review.\n\n"
                + "\n".join(lines) + suffix
            )
        }).encode("utf-8")

        req = urllib.request.Request(
            webhook_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status == 200
    except Exception:
        return False


# --- Weekly Tier-2 verified counts cache (persists across analyze runs) ---
WEEKLY_VERIFIED_CACHE_FILE = Path(__file__).with_name("weekly_verified_counts_cache.json")

def _load_weekly_verified_cache():
    # Session mirror survives ephemeral-disk wipes within a single user session.
    file_data = {}
    try:
        if WEEKLY_VERIFIED_CACHE_FILE.exists():
            with open(WEEKLY_VERIFIED_CACHE_FILE, "r", encoding="utf-8") as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    file_data = loaded
    except Exception as e:
        try:
            st.session_state['weekly_cache_load_error'] = repr(e)
        except Exception:
            pass
    session_mirror = {}
    try:
        session_mirror = dict(st.session_state.get('weekly_verified_cache_mirror', {}) or {})
    except Exception:
        session_mirror = {}
    merged = dict(file_data)
    merged.update(session_mirror)  # session wins on conflict
    return merged

def _save_weekly_verified_counts(year, week, severe_count, scam_count, comment_count, score, model_id):
    key = f"{int(year)}-W{int(week):02d}"
    entry = {
        "severe_count": int(severe_count),
        "scam_count": int(scam_count),
        "comment_count": int(comment_count),
        "score": float(score) if score is not None else None,
        "model_id": str(model_id),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    # Session mirror first — always succeeds, survives disk wipe within session.
    try:
        mirror = dict(st.session_state.get('weekly_verified_cache_mirror', {}) or {})
        mirror[key] = entry
        st.session_state['weekly_verified_cache_mirror'] = mirror
    except Exception:
        pass
    # Best-effort disk write for cross-session persistence when disk is writable.
    try:
        cache = {}
        if WEEKLY_VERIFIED_CACHE_FILE.exists():
            try:
                with open(WEEKLY_VERIFIED_CACHE_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        cache = loaded
            except Exception:
                cache = {}
        cache[key] = entry
        with open(WEEKLY_VERIFIED_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        try:
            st.session_state['weekly_cache_save_error'] = repr(e)
        except Exception:
            pass


def create_dual_trend_plot(selected_week, selected_year, selected_week_data, model_id):
    engine, snowflake_available = get_snowflake_engine()
    if not snowflake_available:
        fig = go.Figure()
        fig.add_annotation(text="Trend data requires database connection", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    weeks_to_display =[]
    selected_year_int = int(selected_year)
    try: start_date_of_selected_week, _ = get_week_dates(selected_year_int, selected_week)
    except ValueError: return go.Figure()

    for offset in[3, 2, 1]:
        target_date = start_date_of_selected_week - timedelta(weeks=offset)
        year, week, _ = target_date.isocalendar()
        weeks_to_display.append((year, week))
    
    week_labels, health_scores, severe_counts, comment_counts, scam_counts = [], [], [], [], []
    trend_status = st.empty()
    trend_status.info(":material/sync: Analyzing 3-week trend history... This may take a moment.")

    verified_cache = _load_weekly_verified_cache()

    for i, (year, week) in enumerate(weeks_to_display):
        df = load_data_for_week(week, year, show_progress=False)
        cached = verified_cache.get(f"{int(year)}-W{int(week):02d}")
        if not df.empty:
            df_em = analyze_emotions_cached(df, model_id=model_id)
            avg = df_em.mean()
            h_score = compute_health_score(avg)

            health_scores.append(h_score if h_score is not None else 0)
            # Tier-2/3 verified counts require Gemini; per perf guardrail we do NOT run LLMs on
            # historical render paths. Fill from the persisted weekly-verified cache when the
            # user has previously analyzed that week; otherwise leave the bar empty.
            if cached:
                severe_counts.append(cached.get('severe_count'))
                scam_counts.append(cached.get('scam_count'))
            else:
                severe_counts.append(None)
                scam_counts.append(None)
            comment_counts.append(len(df))
        else:
            # No data for this week: still surface cached counts if we have them.
            if cached:
                health_scores.append(cached.get('score') if cached.get('score') is not None else 0)
                severe_counts.append(cached.get('severe_count'))
                scam_counts.append(cached.get('scam_count'))
                comment_counts.append(cached.get('comment_count', 0))
            else:
                health_scores.append(0)
                severe_counts.append(None)
                scam_counts.append(None)
                comment_counts.append(0)
        
        start_date, end_date = get_week_dates(year, week)
        if start_date.month == end_date.month:
            date_range = f"{start_date.strftime('%b')} {start_date.day}\u2013{end_date.day}"
        else:
            date_range = f"{start_date.strftime('%b %d')}\u2013{end_date.strftime('%b %d')}"
        week_labels.append(f"Week {week}<br><span style='font-size:11px;color:#555'>{date_range}</span>")

    health_scores.append(selected_week_data['score'])
    severe_counts.append(selected_week_data['severe_count'])
    scam_counts.append(selected_week_data['scam_count'])
    comment_counts.append(selected_week_data['comment_count'])
    
    start_date, end_date = get_week_dates(selected_year_int, selected_week)
    if start_date.month == end_date.month:
        date_range = f"{start_date.strftime('%b')} {start_date.day}\u2013{end_date.day}"
    else:
        date_range = f"{start_date.strftime('%b %d')}\u2013{end_date.strftime('%b %d')}"
    week_labels.append(f"Week {selected_week}<br><span style='font-size:11px;color:#555'>{date_range}</span>")
    trend_status.empty()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Bar(x=week_labels, y=health_scores, name='Health Score', marker_color='rgba(30, 144, 255, 0.6)', text=[f"<b>{s:.1f}</b><br>({c} interactions)" for s, c in zip(health_scores, comment_counts)], textposition='outside', textfont=dict(color='black', weight='bold')), secondary_y=False)
    fig.add_trace(go.Bar(x=week_labels, y=severe_counts, name='Severe Flags (Tier-2 Verified)', marker=dict(color='crimson', line=dict(color='black', width=1.5)), text=[f"<b>{s}</b>" if (s is not None and s > 0) else "" for s in severe_counts], textposition='outside', textfont=dict(color='black', size=12, weight='bold')), secondary_y=True)
    fig.add_trace(go.Bar(x=week_labels, y=scam_counts, name='Scam Flags (Tier-2 Verified)', marker=dict(color='rgba(255, 215, 0, 0.7)', line=dict(color='black', width=1)), text=[f"<b>{s}</b>" if (s is not None and s > 0) else "" for s in scam_counts], textposition='outside', textfont=dict(color='black', size=12, weight='bold')), secondary_y=True)

    fig.update_layout(
        title_text=f"4-Week Health Score (trend) vs. Verified Flagged Items (current week)", 
        title_font=dict(size=13), 
        barmode='group', 
        legend=dict(orientation="h", yanchor="top", y=-0.35, xanchor="center", x=0.5), 
        template="plotly_white", 
        height=500, 
        margin=dict(t=50, b=140, l=50, r=50)
    )
    
    max_secondary = max(
        max((s for s in severe_counts if s is not None), default=0),
        max((s for s in scam_counts if s is not None), default=0),
    )
    fig.update_yaxes(title_text="<b>Flagged Items Count</b>", secondary_y=False, range=[0, 10.5])
    fig.update_yaxes(title_text="", secondary_y=True, range=[0, max_secondary * 1.2 + 5], showgrid=True, gridcolor='lightgrey', gridwidth=1)
    return fig

def create_radar_plot(avg):
    categories = get_emotion_categories()
    negative_emotions = categories[:12]
    if avg.empty:
        fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        return fig
    
    values =[avg.get(c, 0) for c in categories] + [avg.get(categories[0], 0)]
    angles =[n / float(len(categories)) * 2 * pi for n in range(len(categories))] + [0]
    
    fig, ax = plt.subplots(figsize=(7.2, 7.2), subplot_kw=dict(polar=True))
    neg_values =[avg.get(c, 0) for c in negative_emotions] +[avg.get(negative_emotions[0], 0)]
    neg_angles =[n / float(len(categories)) * 2 * pi for n in range(len(negative_emotions))] + [0]
    ax.plot(neg_angles, neg_values, linewidth=2, color='red', alpha=0.8, label='Negative Emotions')
    ax.fill(neg_angles, neg_values, alpha=0.2, color='red')
    
    neutral_idx = len(negative_emotions)
    neutral_angle = neutral_idx / float(len(categories)) * 2 * pi
    neutral_value = avg.get('neutral', 0)
    ax.plot([neutral_angle, neutral_angle], [0, neutral_value], linewidth=3, color='black', label='Neutral', alpha=0.9)
    
    positive_emotions = categories[13:]
    pos_start_idx = len(negative_emotions) + 1
    pos_values =[avg.get(c, 0) for c in positive_emotions] +[avg.get(positive_emotions[0], 0)]
    pos_angles =[n / float(len(categories)) * 2 * pi for n in range(pos_start_idx, len(categories))] +[pos_start_idx / float(len(categories)) * 2 * pi]
    ax.plot(pos_angles, pos_values, linewidth=2, color='blue', alpha=0.8, label='Positive Emotions')
    ax.fill(pos_angles, pos_values, alpha=0.2, color='blue')
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9)
    # Use only the polygon emotion data (not neutral) to set scale,
    # so that the neutral spoke (often 0.4-0.5) does not crush the
    # negative and positive polygon areas into invisibility.
    polygon_max = max(
        max((v for v in neg_values[:-1]), default=0.0),
        max((v for v in pos_values[:-1]), default=0.0),
    )
    ax.set_ylim(0, max(polygon_max * 1.25, 0.12))
    ax.set_title("Community Emotion Profile", size=15, fontweight='bold', pad=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, frameon=False)
    fig.subplots_adjust(top=0.88, bottom=0.05, left=0.05, right=0.95)
    return fig

def create_emotion_trend_chart(selected_week, selected_year, model_id):
    engine, snowflake_available = get_snowflake_engine()
    if not snowflake_available:
        fig = go.Figure()
        fig.add_annotation(text="Emotion trend requires database connection", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
    
    df_selected = load_data_for_week(selected_week, int(selected_year), show_progress=False)
    if df_selected.empty: return go.Figure()
    
    df_em_selected = analyze_emotions_cached(df_selected, model_id=model_id)
    avg_selected = df_em_selected.mean()
    negative_emotions = get_emotion_categories()[:12]
    neg_intensities = {emotion: avg_selected.get(emotion, 0) for emotion in negative_emotions}
    top_6_emotions = sorted(neg_intensities.items(), key=lambda x: x[1], reverse=True)[:6]
    top_6_emotion_names =[emotion for emotion, _ in top_6_emotions]
    
    weeks_to_display =[]
    selected_year_int = int(selected_year)
    try: start_date_of_selected_week, _ = get_week_dates(selected_year_int, selected_week)
    except ValueError: return go.Figure()

    for offset in[3, 2, 1, 0]:
        target_date = start_date_of_selected_week - timedelta(weeks=offset)
        year, week, _ = target_date.isocalendar()
        weeks_to_display.append((year, week))
    
    emotion_data = {emotion:[] for emotion in top_6_emotion_names}
    week_labels =[]
    
    for year, week in weeks_to_display:
        df = load_data_for_week(week, year, show_progress=False)
        if df.empty:
            for emotion in top_6_emotion_names: emotion_data[emotion].append(0)
        else:
            df_em = analyze_emotions_cached(df, model_id=model_id)
            avg = df_em.mean() if not df_em.empty else pd.Series()
            for emotion in top_6_emotion_names:
                emotion_data[emotion].append(round(avg.get(emotion, 0) * 100, 1))
        week_labels.append(f"Week {week}")
    
    fig = go.Figure()
    distinct_colors =['#FF0000', '#FFD700', '#006400', '#0000FF', '#FF00FF', '#4B0000']
    for i, emotion in enumerate(top_6_emotion_names):
        color = distinct_colors[i % len(distinct_colors)]
        fig.add_trace(go.Scatter(x=week_labels, y=emotion_data[emotion], mode='lines+markers', name=emotion.title(), line=dict(width=3, color=color), marker=dict(size=8)))
        last_value = emotion_data[emotion][-1]
        if last_value > 0:
            fig.add_annotation(x=week_labels[-1], y=last_value, text=emotion.title(), showarrow=False, xshift=40, font=dict(size=10, color=color), bgcolor="rgba(255,255,255,0.8)", bordercolor=color, borderwidth=1)
    
    fig.update_layout(title=f"Top 6 Negative Emotion Trends (Ending Week {selected_week})", xaxis_title="Week", yaxis_title="Emotion Intensity (%)", height=450, template="plotly_white", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), showlegend=True, margin=dict(r=120))
    return fig


def create_fn_rate_trend_chart(fn_rate_history: list) -> go.Figure:
    """
    Line chart of false-negative rate (%) and CRITICAL scam flag count per weekly run.
    Data is accumulated in st.session_state['fn_rate_history'] during this session.
    """
    if not fn_rate_history:
        fig = go.Figure()
        fig.add_annotation(
            text="Run analyses across multiple weeks to see the false-negative trend.",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=13, color="#888")
        )
        fig.update_layout(height=250, template="plotly_white",
                          title="Safety Trend: False-Negative Rate & Critical Scam Flags per Week")
        return fig

    labels        = [h["label"]                     for h in fn_rate_history]
    fn_rates      = [h["false_negative_rate"]        for h in fn_rate_history]
    scam_counts   = [h["scam_flags"]                 for h in fn_rate_history]
    fn_detected   = [h["false_negatives_detected"]   for h in fn_rate_history]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=fn_rates,
        mode="lines+markers", name="FN Rate (%)",
        line=dict(color="#d32d27", width=2),
        marker=dict(size=7),
        yaxis="y1",
        hovertemplate="Week: %{x}<br>FN Rate: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=labels, y=scam_counts,
        name="Scam Flags (confirmed)",
        marker_color="#00387b", opacity=0.55,
        yaxis="y2",
        hovertemplate="Week: %{x}<br>Scam Flags: %{y}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=fn_detected,
        mode="lines+markers", name="Missed Scams (sampled)",
        line=dict(color="#ff9f1a", width=1.5, dash="dot"),
        marker=dict(size=6, symbol="diamond"),
        yaxis="y1",
        hovertemplate="Week: %{x}<br>Missed (sample): %{y}<extra></extra>",
    ))

    fig.update_layout(
        title="Safety Trend: False-Negative Rate & Scam Flags per Weekly Run",
        xaxis_title="Week",
        yaxis=dict(title="FN Rate / Missed Count", side="left", showgrid=True),
        yaxis2=dict(title="Confirmed Scam Flags", side="right", overlaying="y", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=310, template="plotly_white",
        margin=dict(t=60, b=30),
    )
    return fig


def extract_week_number(week_str):
    if isinstance(week_str, str) and "Week" in week_str: return int(week_str.split()[1])
    return int(week_str)


@st.cache_data(show_spinner=False, ttl=24*60*60)
def _get_build_sha():
    """Resolve current git short SHA so the sidebar badge proves the running code matches HEAD."""
    repo_root = Path(__file__).resolve().parent
    head_file = repo_root / ".git" / "HEAD"
    try:
        head = head_file.read_text(encoding="utf-8").strip()
    except Exception:
        return ("unknown", "no .git")
    if head.startswith("ref:"):
        ref_path = repo_root / ".git" / head.split(" ", 1)[1].strip()
        try:
            sha = ref_path.read_text(encoding="utf-8").strip()
            return (sha[:7], "HEAD")
        except Exception:
            packed = repo_root / ".git" / "packed-refs"
            try:
                for line in packed.read_text(encoding="utf-8").splitlines():
                    if line.endswith(head.split(" ", 1)[1].strip()):
                        return (line.split()[0][:7], "HEAD")
            except Exception:
                pass
            return ("unknown", "ref unresolved")
    return (head[:7], "detached")

def get_week_choices(selected_year):
    weeks = []
    for w in range(1, 54):  # Allow week 53 for leap years
        try:
            week_start = datetime.strptime(f'{selected_year}-W{w:02d}-1', "%G-W%V-%u")
            month_name = week_start.strftime("%b")
            weeks.append(f"Week {w} ({month_name})")
        except ValueError:
            pass # Year doesn't have a week 53
    return weeks

@st.dialog("📝 Sanity Check Details")
def view_comment_details(entry):
    st.markdown(f"**Item ID:** {entry.get('comment_index', 'N/A')}")
    st.markdown(f"**Timestamp:** {entry.get('timestamp', 'N/A')}")
    st.markdown("---")
    st.subheader("Full Text")
    full_text = entry.get('full_comment', entry.get('comment_preview', 'No text available.'))
    st.text_area("Message Content", value=full_text, height=300, disabled=True)
    st.markdown("---")
    st.subheader("🤖 Gemini Analysis")
    st.code(entry.get('raw_response', 'No analysis available.'), language="text")
    verdict = "✅ VERIFIED" if entry.get('interpreted_label') is True else "❌ REJECTED"
    if entry.get('status') == 'Ambiguous': verdict = "❔ UNCERTAIN"
    st.markdown(f"### Final Verdict: {verdict}")

def display_sanity_check_log():
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sanity-Check Call Log (Cumulative)**")
    st.sidebar.caption("💡 Click a row to see full text & reasoning (Chain-of-Thought)")
    
    if st.sidebar.button("Clear Log History", icon=":material/delete:"):
        st.session_state.sanity_log =[]
        st.rerun()

    if "sanity_log" in st.session_state and st.session_state.sanity_log:
        try:
            if 'status' not in st.session_state.sanity_log[0]:
                 st.session_state.sanity_log =[]
                 st.rerun()
        except: st.session_state.sanity_log =[]

    if "sanity_log" in st.session_state and st.session_state.sanity_log:
        log_entries = st.session_state.sanity_log
        display_entries =[dict(log) for log in log_entries]
        cleaned_rows =[]
        for log in display_entries:
            status = log.get('status', 'Unknown')
            verdict_str = ""
            if status == 'Separator': verdict_str = "---"
            elif "Error" in status or status == 'Ambiguous': verdict_str = "❔ UNCERTAIN"
            elif log.get('interpreted_label') is True: verdict_str = "✅ VERIFIED"
            elif log.get('interpreted_label') is False: verdict_str = "❌ REJECTED"
            else: verdict_str = "❌ REJECTED"

            cleaned_rows.append({
                "Timestamp": str(log.get('timestamp', '')),
                "Item #": str(log.get('comment_index', '')),
                "Gemini Verdict": str(verdict_str),
                "Flagged For": str(log.get('flagged_for', '')),
                "Text Preview": str(log.get('comment_preview', ''))
            })
        
        try:
            df_log = pd.DataFrame(cleaned_rows).astype(str)
            def style_verdict(verdict):
                if not isinstance(verdict, str): return ''
                if "---" in verdict: return 'background-color: #e9ecef; color: #495057; font-weight: bold;'
                if "VERIFIED" in verdict: return 'background-color: #D4EDDA; color: #155724'
                if "REJECTED" in verdict: return 'background-color: #F8D7DA; color: #721C24'
                return 'background-color: #FFF3CD; color: #856404'

            if not df_log.empty:
                styled_df = df_log.style.map(style_verdict, subset=['Gemini Verdict'])
                selection = st.sidebar.dataframe(styled_df, column_order=("Timestamp", "Item #", "Gemini Verdict", "Flagged For", "Text Preview"), use_container_width=True, height=400, on_select="rerun", selection_mode="single-row")
                if selection.selection.rows:
                    selected_idx = selection.selection.rows[0]
                    if selected_idx < len(log_entries):
                        selected_entry = log_entries[selected_idx]
                        if selected_entry.get('status') != 'Separator': view_comment_details(selected_entry)
                        else: st.sidebar.info("This is a separator line.")
        except Exception as e:
            st.sidebar.error(f"Log display error: {e}")
    else:
        st.sidebar.info("No sanity-check calls have been made in this session.")

def _gemini_distillation_call(comment_text):
    teacher_model, available = load_gemini_model(TEACHER_MODEL_NAME)
    if not available:
        teacher_model, available = load_gemini_model(FAST_REVIEW_MODEL_NAME)
    if not available:
        return "ERROR: Model unavailable"
    prompt = DISTILLATION_PROMPT_TEMPLATE.format(comment_text=comment_text)
    try:
        resp = generate_with_retry(teacher_model, prompt, generation_config=get_teacher_generation_config())
        return extract_json_from_text(resp.text or "")
    except Exception as e:
        return f"ERROR: {e}"

def calculate_entropy(row):
    """Calculates Shannon Entropy for Active Learning Uncertainty Sampling"""
    probs = np.array([float(row.get(f'student_{e}', 0.0)) for e in EMOTION_LABELS])
    probs = np.clip(probs, 1e-9, 1.0) # Prevent log(0)
    probs /= probs.sum() # Normalize to sum to 1
    return -np.sum(probs * np.log(probs))

def generate_distillation_data(df, df_em, sample_size=60, excluded_hashes=None):
    if df.empty or df_em.empty:
        st.warning("No data available to sample.")
        return None

    working_df = df.reset_index(drop=True).copy()
    working_em = df_em.reset_index(drop=True).copy()
    working_df['content'] = working_df['content'].astype(str).apply(clean_text)
    working_df['normalized_text'] = working_df['content'].apply(normalize_training_text)
    working_df['text_hash'] = working_df['content'].apply(build_text_hash)
    working_df['char_len'] = working_df['content'].str.len()

    working = pd.concat([working_df, working_em.add_prefix('student_')], axis=1)
    working = working[working['content'].apply(is_viable_training_text)].copy()
    working = working[working['normalized_text'] != ''].copy()
    working = working.drop_duplicates(subset=['text_hash']).reset_index(drop=True)

    excluded_hashes = set(excluded_hashes or [])
    if excluded_hashes:
        before_exclusion = len(working)
        working = working[~working['text_hash'].isin(excluded_hashes)].reset_index(drop=True)
        excluded_count = before_exclusion - len(working)
        if excluded_count > 0:
            st.caption(f"Excluded {excluded_count:,} texts already present in the master dataset.")

    if len(working) < 10:
        st.warning("Not enough high-quality unique text data to sample.")
        return None

    requested_total = max(1, int(sample_size))
    target_total = min(requested_total, len(working))
    if target_total < requested_total:
        st.info(f"Requested {requested_total:,} samples, but only {len(working):,} unique viable texts were available. Using {target_total:,}.")

    thresholds = get_label_thresholds()
    for emotion in EMOTION_LABELS:
        student_col = f'student_{emotion}'
        if student_col not in working.columns:
            working[student_col] = 0.0
        working[student_col] = pd.to_numeric(working[student_col], errors='coerce').fillna(0.0).clip(0.0, 1.0)

    # --- Shannon Entropy Calculation for Uncertainty Sampling ---
    working['entropy'] = working.apply(calculate_entropy, axis=1)
    entropy_threshold = working['entropy'].quantile(0.75) # Target the top 25% most uncertain items

    neg_cols = [f'student_{emotion}' for emotion in NEGATIVE_EMOTIONS if emotion != 'confusion']
    serious_cols = [f'student_{emotion}' for emotion in SERIOUS_NEGATIVE_EMOTIONS]
    rare_cols = [f'student_{emotion}' for emotion in FOCUS_RARE_EMOTIONS]
    false_positive_cols = [f'student_{emotion}' for emotion in FALSE_POSITIVE_EMOTIONS]

    working['student_negative_sum'] = working[neg_cols].sum(axis=1)
    working['student_serious_spike'] = working[serious_cols].max(axis=1)
    working['student_primary_emotion'] = working[[f'student_{emotion}' for emotion in EMOTION_LABELS]].idxmax(axis=1).str.replace('student_', '', regex=False)
    working['student_primary_score'] = working.apply(lambda row: float(row.get(f"student_{row['student_primary_emotion']}", 0.0)), axis=1)
    working['rare_hint_emotions'] = working['content'].apply(get_rare_hint_emotions)
    working['rare_hint_count'] = working['rare_hint_emotions'].apply(len)
    working['student_rare_signal'] = working[rare_cols].max(axis=1)
    working['student_false_positive_signal'] = working[false_positive_cols].max(axis=1)
    working['conflict_signal'] = working['content'].apply(has_false_positive_conflict_signal)
    working['student_neutral_score'] = pd.to_numeric(working.get('student_neutral', 0.0), errors='coerce').fillna(0.0).clip(0.0, 1.0)
    working['non_neutral_peak'] = working[[f'student_{emotion}' for emotion in EMOTION_LABELS if emotion != 'neutral']].max(axis=1)

    rare_mask = (working['student_rare_signal'] >= 0.08) | (working['rare_hint_count'] > 0)
    uncertain_mask = working['entropy'] >= entropy_threshold # Replaced margin logic with Entropy
    false_positive_mask = (
        (working['student_false_positive_signal'] >= 0.25) &
        (
            (working['student_negative_sum'] >= 0.25) |
            (working['student_serious_spike'] >= 0.20) |
            (working['rare_hint_count'] > 0) |
            (working['conflict_signal'])
        )
    )
    neutral_mask = (
        (working['student_neutral_score'] >= thresholds.get('neutral', 0.25)) &
        (working['non_neutral_peak'] <= 0.18) &
        (~rare_mask)
    )
    random_mask = pd.Series(True, index=working.index)

    bucket_masks = {
        'rare_focus': rare_mask,
        'uncertain': uncertain_mask,
        'false_positive_suspect': false_positive_mask,
        'neutral_baseline': neutral_mask,
        'random_baseline': random_mask,
    }

    bucket_targets = compute_bucket_targets(target_total, PRODUCTION_SAMPLING_PLAN)
    rng = np.random.default_rng()
    selected_indices = []
    bucket_by_index = {}
    bucket_pick_counts = {}

    for bucket_name, _ in PRODUCTION_SAMPLING_PLAN:
        bucket_target = bucket_targets.get(bucket_name, 0)
        available_indices = [idx for idx in working.index[bucket_masks[bucket_name]].tolist() if idx not in selected_indices]
        if bucket_target <= 0 or not available_indices:
            bucket_pick_counts[bucket_name] = 0
            continue
        chosen = rng.choice(available_indices, size=min(bucket_target, len(available_indices)), replace=False).tolist()
        bucket_pick_counts[bucket_name] = len(chosen)
        for idx in chosen:
            selected_indices.append(idx)
            bucket_by_index[idx] = bucket_name

    if len(selected_indices) < target_total:
        remaining_indices = [idx for idx in working.index.tolist() if idx not in selected_indices]
        if remaining_indices:
            filler = rng.choice(remaining_indices, size=min(target_total - len(selected_indices), len(remaining_indices)), replace=False).tolist()
            bucket_pick_counts['random_fill'] = len(filler)
            for idx in filler:
                selected_indices.append(idx)
                bucket_by_index[idx] = 'random_fill'

    sampled_df = working.loc[selected_indices].copy()
    sampled_df['sampler_bucket'] = [bucket_by_index.get(idx, 'random_baseline') for idx in selected_indices]
    sampled_df = sampled_df.reset_index(drop=True)

    st.caption(
        "Production sample plan → " +
        ", ".join(
            f"{bucket}: target {bucket_targets.get(bucket, 0)}, pulled {bucket_pick_counts.get(bucket, 0)}"
            for bucket, _ in PRODUCTION_SAMPLING_PLAN
        ) +
        (f", fill {bucket_pick_counts.get('random_fill', 0)}" if bucket_pick_counts.get('random_fill', 0) else "")
    )

    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.markdown(
        f":material/science: **Teacher-label generation:** Gemini scoring {len(sampled_df):,} production samples using Shannon Entropy Uncertainty Sampling..."
    )
    progress_bar.progress(0.0)

    comments_to_process = sampled_df['content'].astype(str).tolist()
    raw_results = process_in_parallel_batched(
        _gemini_distillation_call,
        comments_to_process,
        progress_bar,
        status_text,
        ":material/science: **Teacher-label generation:** Gemini scored {completed:,}/{total:,} items",
        max_workers=10,
        batch_size=PRODUCTION_BATCH_SIZE
    )

    for idx, comment in enumerate(comments_to_process):
        raw_text = raw_results.get(idx, "ERROR: Timeout")
        if raw_text.startswith("ERROR:"):
            continue
        try:
            scores = normalize_emotion_scores(json.loads(raw_text))
            active_labels = get_active_labels_from_scores(scores, thresholds)
            source_row = sampled_df.iloc[idx]
            row_data = {
                'text': comment,
                'labels': ', '.join(active_labels),
                'text_hash': source_row.get('text_hash'),
                'sampler_bucket': source_row.get('sampler_bucket', 'random_baseline'),
                'activity_type': source_row.get('activity_type', 'Unknown'),
                'community_name': source_row.get('community_name', 'N/A'),
                'source_commenter_name': source_row.get('commenter_name', 'N/A'),
                'student_primary_emotion': source_row.get('student_primary_emotion', 'neutral'),
                'student_primary_score': round(float(source_row.get('student_primary_score', 0.0)), 4),
                'student_negative_sum': round(float(source_row.get('student_negative_sum', 0.0)), 4),
                'student_false_positive_signal': round(float(source_row.get('student_false_positive_signal', 0.0)), 4),
                'student_entropy': round(float(source_row.get('entropy', 0.0)), 4),
                'rare_hint_emotions': ', '.join(source_row.get('rare_hint_emotions', [])),
                'teacher_data_version': DISTILLATION_DATA_VERSION,
                'teacher_default_threshold': 0.4,
            }
            for emotion in EMOTION_LABELS:
                row_data[emotion] = scores.get(emotion, 0.0)
            results.append(row_data)
        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()

    result_df = pd.DataFrame(results)
    st.session_state['last_generation_stats'] = {
        'sampled_candidates': int(len(sampled_df)),
        'teacher_valid_rows': int(len(result_df)),
        'teacher_invalid_or_failed': int(max(0, len(sampled_df) - len(result_df))),
    }

    if result_df.empty:
        st.error("Teacher labeling completed, but no valid labeled rows were produced. This usually means the model responses were invalid, timed out, or did not parse as JSON.")
        return pd.DataFrame()

    return result_df


# ==========================================
# --- EXTERNAL DATA AUGMENTATION HELPERS ---
# ==========================================
EXTERNAL_ED_CONTEXT_MAP = {
    "afraid": {"fear": 1.0},
    "angry": {"anger": 1.0},
    "annoyed": {"annoyance": 1.0},
    "anticipating": {"optimism": 0.8, "curiosity": 0.4},
    "anxious": {"nervousness": 1.0},
    "apprehensive": {"nervousness": 0.9, "fear": 0.5},
    "ashamed": {"embarrassment": 0.8, "remorse": 0.6},
    "caring": {"caring": 1.0},
    "confident": {"optimism": 0.6, "pride": 0.6},
    "content": {"joy": 0.6, "relief": 0.4},
    "devastated": {"grief": 0.7, "sadness": 1.0},
    "disappointed": {"disappointment": 1.0},
    "disgusted": {"disgust": 1.0},
    "embarrassed": {"embarrassment": 1.0},
    "excited": {"excitement": 1.0},
    "faithful": {"love": 0.4, "approval": 0.4},
    "furious": {"anger": 1.0},
    "grateful": {"gratitude": 1.0},
    "guilty": {"remorse": 1.0},
    "hopeful": {"optimism": 1.0},
    "impressed": {"admiration": 0.8, "approval": 0.4},
    "jealous": {"disapproval": 0.4, "annoyance": 0.3},
    "joyful": {"joy": 1.0},
    "lonely": {"sadness": 0.8, "grief": 0.3},
    "nostalgic": {"realization": 0.4, "sadness": 0.3, "joy": 0.3},
    "prepared": {"optimism": 0.4},
    "proud": {"pride": 1.0},
    "sad": {"sadness": 1.0},
    "sentimental": {"love": 0.4, "sadness": 0.4, "grief": 0.2},
    "surprised": {"surprise": 1.0},
    "terrified": {"fear": 1.0},
    "trusting": {"approval": 0.5, "caring": 0.3},
}

EXTERNAL_ISEAR_MAP = {
    "joy": {"joy": 1.0},
    "fear": {"fear": 1.0},
    "anger": {"anger": 1.0},
    "sadness": {"sadness": 1.0},
    "disgust": {"disgust": 1.0},
    "shame": {"embarrassment": 0.8, "remorse": 0.4},
    "guilt": {"remorse": 1.0},
}

EXTERNAL_GO_RARE_LABELS = ['grief', 'remorse', 'relief', 'pride', 'fear', 'nervousness', 'realization']
EXTERNAL_DATASET_DEFAULT_WEIGHTS = {
    'go_emotions_rare': 0.55,
    'empathetic_dialogues_contexts': 0.70,
    'isear': 0.75,
    'counsel_chat_teacher': 0.55,
}

def build_goemotions_rare_supplement(go_df, rare_cap_per_label=60):
    if go_df is None or go_df.empty:
        return pd.DataFrame()
    rare_cols = [c for c in EXTERNAL_GO_RARE_LABELS if c in go_df.columns]
    if not rare_cols:
        return pd.DataFrame()
    rare_df = go_df[go_df[rare_cols].sum(axis=1) > 0].copy()
    if rare_df.empty:
        return rare_df
    kept = []
    for emotion in rare_cols:
        subset = rare_df[rare_df[emotion] > 0].head(rare_cap_per_label)
        if not subset.empty:
            kept.append(subset)
    if not kept:
        return pd.DataFrame()
    out = pd.concat(kept, ignore_index=True).drop_duplicates(subset=['text_hash']).reset_index(drop=True)
    out['source_dataset'] = 'go_emotions_rare'
    out['sample_weight'] = EXTERNAL_DATASET_DEFAULT_WEIGHTS['go_emotions_rare']
    out['teacher_data_version'] = 'external_go_rare_v2'
    out['sampler_bucket'] = 'external_go_rare'
    return _finalize_external_df(out)

def external_append_preview_df(include_go_rare=False, include_mapped=False, go_cap_per_label=60):
    frames = []
    if include_go_rare and os.path.exists('exact_append_ready.csv'):
        go_df = pd.read_csv('exact_append_ready.csv')
        rare_df = build_goemotions_rare_supplement(go_df, rare_cap_per_label=go_cap_per_label)
        if not rare_df.empty:
            frames.append(rare_df)
    if include_mapped and os.path.exists('mapped_append_ready.csv'):
        mapped_df = pd.read_csv('mapped_append_ready.csv')
        if not mapped_df.empty:
            mapped_df = mapped_df.copy()
            mapped_df['sample_weight'] = mapped_df['source_dataset'].map(EXTERNAL_DATASET_DEFAULT_WEIGHTS).fillna(mapped_df.get('sample_weight', 0.7))
            mapped_df['teacher_data_version'] = mapped_df.get('teacher_data_version', 'external_mapped_v2')
            frames.append(mapped_df)

    if not frames:
        return pd.DataFrame()

    combined = _finalize_external_df(pd.concat(frames, ignore_index=True))
    if combined.empty:
        return combined

    if os.path.exists('master_training_data.csv'):
        existing_df = pd.read_csv('master_training_data.csv')
        if 'text_hash' not in existing_df.columns and 'text' in existing_df.columns:
            existing_df['text_hash'] = existing_df['text'].astype(str).apply(build_text_hash)
        existing_hashes = set(existing_df['text_hash'].dropna().astype(str).tolist()) if ('text_hash' in existing_df.columns and not existing_df.empty) else set()
        combined = combined[~combined['text_hash'].isin(existing_hashes)].reset_index(drop=True)

    return combined

def build_external_preview_summary(preview_df):
    if preview_df is None or preview_df.empty:
        return {
            'preview_rows': 0,
            'source_counts': {},
            'weight_counts': {},
            'rare_label_counts': {},
        }
    rare_labels = [label for label in EXTERNAL_GO_RARE_LABELS if label in preview_df.columns]
    return {
        'preview_rows': int(len(preview_df)),
        'source_counts': preview_df['source_dataset'].value_counts().to_dict() if 'source_dataset' in preview_df.columns else {},
        'weight_counts': preview_df['sample_weight'].astype(str).value_counts().to_dict() if 'sample_weight' in preview_df.columns else {},
        'rare_label_counts': {label: int((preview_df[label] > 0).sum()) for label in rare_labels},
    }

def _clear_external_fetch_diagnostics():
    st.session_state['external_fetch_diagnostics'] = []

def _record_external_fetch_diag(source_name, message):
    st.session_state.setdefault('external_fetch_diagnostics', [])
    st.session_state['external_fetch_diagnostics'].append(f"{source_name}: {message}")

def _require_datasets_library():
    try:
        from datasets import load_dataset
        return load_dataset
    except Exception as e:
        st.error("The Hugging Face `datasets` package is required for external augmentation. Install it in the app environment first.")
        st.caption(f"Import error: {e}")
        _record_external_fetch_diag('datasets', f'import failed -> {e}')
        return None


def has_datasets_library():
    """Lightweight availability check for optional external-data augmentation dependency."""
    try:
        import datasets  # noqa: F401
        return True
    except Exception:
        return False

def _empty_external_label_row():
    return {emotion: 0.0 for emotion in EMOTION_LABELS}

def _finalize_external_df(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['text'] = df['text'].astype(str).apply(clean_text)
    df['text'] = df['text'].astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    df = df[df['text'].str.len() >= 12].copy()
    df['text_hash'] = df['text'].astype(str).apply(build_text_hash)
    df = df.drop_duplicates(subset=['text_hash']).reset_index(drop=True)
    return df

def _build_external_row(text, labels, source_dataset, label_origin, mapping_confidence, sample_weight, external_label=""):
    row = {
        'text': clean_text(text),
        'labels': '',
        'text_hash': build_text_hash(text),
        'sampler_bucket': f"external_{label_origin}",
        'activity_type': 'External',
        'community_name': source_dataset,
        'source_commenter_name': 'External Dataset',
        'student_primary_emotion': 'external_seed',
        'student_primary_score': 0.0,
        'student_negative_sum': 0.0,
        'student_false_positive_signal': 0.0,
        'min_threshold_margin': 0.0,
        'rare_hint_emotions': '',
        'teacher_data_version': f"external_{label_origin}_v1",
        'teacher_default_threshold': 0.4,
        'source_dataset': source_dataset,
        'label_origin': label_origin,
        'mapping_confidence': mapping_confidence,
        'sample_weight': float(sample_weight),
        'external_label': external_label,
    }
    row.update(_empty_external_label_row())
    for emotion, score in labels.items():
        if emotion in row:
            row[emotion] = float(score)
    active_labels = [emotion for emotion in EMOTION_LABELS if float(row.get(emotion, 0.0)) > 0]
    row['labels'] = ', '.join(active_labels)
    return row

def _cap_external_per_label(df, cap):
    if df.empty:
        return df
    kept = []
    for emotion in EMOTION_LABELS:
        subset = df[df[emotion] > 0].head(cap)
        if not subset.empty:
            kept.append(subset)
    if not kept:
        return df
    return pd.concat(kept, ignore_index=True).drop_duplicates(subset=['text_hash']).reset_index(drop=True)


def _extract_goemotions_rows_from_datasetdict(ds):
    frames = []
    label_names = None

    for split_name in ['train', 'validation', 'test']:
        if split_name in ds:
            split = ds[split_name]
            try:
                labels_feature = split.features['labels']
                if hasattr(labels_feature, 'feature') and hasattr(labels_feature.feature, 'names'):
                    label_names = list(labels_feature.feature.names)
                    break
                if hasattr(labels_feature, 'names'):
                    label_names = list(labels_feature.names)
                    break
            except Exception:
                continue

    for split_name in ['train', 'validation', 'test']:
        if split_name in ds:
            split_df = pd.DataFrame(ds[split_name])
            frames.append(split_df)
    if not frames:
        first_key = next(iter(ds.keys()))
        frames = [pd.DataFrame(ds[first_key])]
    df = pd.concat(frames, ignore_index=True)

    text_col = 'text' if 'text' in df.columns else None
    if text_col is None:
        for col in df.columns:
            if str(col).lower() == 'text':
                text_col = col
                break
    if text_col is None:
        raise ValueError(f'GoEmotions text column not found. Columns: {df.columns.tolist()}')

    if all(emotion in df.columns for emotion in EMOTION_LABELS):
        rows = []
        for _, r in df.iterrows():
            text_value = clean_text(r[text_col])
            if not text_value:
                continue
            labels = {emotion: float(r.get(emotion, 0.0)) for emotion in EMOTION_LABELS}
            if sum(labels.values()) <= 0:
                continue
            rows.append(_build_external_row(
                text=text_value,
                labels=labels,
                source_dataset='go_emotions',
                label_origin='exact',
                mapping_confidence='exact',
                sample_weight=1.0,
                external_label='multi_label_exact'
            ))
        return _finalize_external_df(pd.DataFrame(rows))

    if 'labels' not in df.columns or not label_names:
        raise ValueError(
            'GoEmotions schema is unsupported in this runtime. Expected either one emotion column per label or '
            'a labels feature with class names.'
        )

    rows = []
    valid_names = set(EMOTION_LABELS)
    for _, r in df.iterrows():
        text_value = clean_text(r[text_col])
        if not text_value:
            continue
        labels = {emotion: 0.0 for emotion in EMOTION_LABELS}
        raw_labels = r['labels']
        if raw_labels is None or (isinstance(raw_labels, float) and pd.isna(raw_labels)):
            continue
        if not isinstance(raw_labels, (list, tuple)):
            raw_labels = [raw_labels]
        for idx in raw_labels:
            try:
                idx_int = int(idx)
            except Exception:
                continue
            if 0 <= idx_int < len(label_names):
                label_name = str(label_names[idx_int]).lower()
                if label_name in valid_names:
                    labels[label_name] = 1.0
        if sum(labels.values()) <= 0:
            continue
        rows.append(_build_external_row(
            text=text_value,
            labels=labels,
            source_dataset='go_emotions',
            label_origin='exact',
            mapping_confidence='exact',
            sample_weight=1.0,
            external_label='multi_label_exact'
        ))
    return _finalize_external_df(pd.DataFrame(rows))

@st.cache_data(show_spinner=False)
def fetch_goemotions_exact_rows(max_rows=1200):
    load_dataset = _require_datasets_library()
    if load_dataset is None:
        return pd.DataFrame()
    candidates = [
        ("google-research-datasets/go_emotions", "raw"),
        ("google-research-datasets/go_emotions", "simplified"),
        ("go_emotions", "raw"),
        ("go_emotions", None),
        ("rico2512/go_emotions", None),
    ]
    ds = None
    last_err = None
    for name, cfg in candidates:
        try:
            ds = load_dataset(name, cfg) if cfg else load_dataset(name)
            _record_external_fetch_diag('GoEmotions', f'loaded {name} config={cfg}')
            break
        except Exception as e:
            last_err = e
    if ds is None:
        _record_external_fetch_diag('GoEmotions', f'load failed -> {last_err}')
        return pd.DataFrame()

    try:
        out = _extract_goemotions_rows_from_datasetdict(ds)
    except Exception as e:
        _record_external_fetch_diag('GoEmotions', f'parse failed -> {e}')
        return pd.DataFrame()

    if max_rows and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def fetch_ed_mapped_rows(max_rows=400):
    load_dataset = _require_datasets_library()
    if load_dataset is None:
        return pd.DataFrame()
    try:
        ds = load_dataset("bdotloh/empathetic-dialogues-contexts")
    except Exception as e:
        _record_external_fetch_diag('EmpatheticDialogues', f'load failed -> {e}')
        return pd.DataFrame()

    frames = []
    for split_name in ['train', 'validation', 'valid', 'test']:
        if split_name in ds:
            frames.append(pd.DataFrame(ds[split_name]))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(ds[next(iter(ds.keys()))])

    text_col = next((c for c in ['context', 'text', 'utterance', 'situation'] if c in df.columns), None)
    label_col = next((c for c in ['emotion', 'label'] if c in df.columns), None)
    if text_col is None or label_col is None:
        _record_external_fetch_diag('EmpatheticDialogues', f'schema mismatch -> columns={df.columns.tolist()}')
        return pd.DataFrame()

    rows = []
    for _, r in df.iterrows():
        ext_label = str(r[label_col]).strip().lower()
        if ext_label not in EXTERNAL_ED_CONTEXT_MAP:
            continue
        rows.append(_build_external_row(
            text=r[text_col],
            labels=EXTERNAL_ED_CONTEXT_MAP[ext_label],
            source_dataset='empathetic_dialogues_contexts',
            label_origin='mapped',
            mapping_confidence='medium',
            sample_weight=0.7,
            external_label=ext_label
        ))
    out = _finalize_external_df(pd.DataFrame(rows))
    if max_rows and len(out) > max_rows:
        out = _cap_external_per_label(out, max(40, max_rows // 12))
        if len(out) > max_rows:
            out = out.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def fetch_isear_mapped_rows(max_rows=400):
    load_dataset = _require_datasets_library()
    if load_dataset is None:
        return pd.DataFrame()
    try:
        ds = load_dataset("savalera/isear-from-original", "filtered")
    except Exception as e:
        _record_external_fetch_diag('ISEAR', f'load failed -> {e}')
        return pd.DataFrame()

    frames = []
    for split_name in ['train', 'validation', 'test']:
        if split_name in ds:
            frames.append(pd.DataFrame(ds[split_name]))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(ds[next(iter(ds.keys()))])

    text_col = next((c for c in ['SIT', 'text', 'situation'] if c in df.columns), None)
    label_col = next((c for c in ['emotion', 'EMOT', 'label'] if c in df.columns), None)
    if text_col is None or label_col is None:
        _record_external_fetch_diag('ISEAR', f'schema mismatch -> columns={df.columns.tolist()}')
        return pd.DataFrame()

    emot_to_name = {1: "joy", 2: "fear", 3: "anger", 4: "sadness", 5: "disgust", 6: "shame", 7: "guilt"}

    rows = []
    for _, r in df.iterrows():
        raw_label = r[label_col]
        if isinstance(raw_label, (int, float)) and not pd.isna(raw_label):
            ext_label = emot_to_name.get(int(raw_label), "").lower()
        else:
            ext_label = str(raw_label).strip().lower()
        if ext_label not in EXTERNAL_ISEAR_MAP:
            continue
        confidence = 'medium' if ext_label in {'shame', 'guilt'} else 'high'
        weight = 0.7 if confidence == 'medium' else 0.85
        rows.append(_build_external_row(
            text=r[text_col],
            labels=EXTERNAL_ISEAR_MAP[ext_label],
            source_dataset='isear',
            label_origin='mapped',
            mapping_confidence=confidence,
            sample_weight=weight,
            external_label=ext_label
        ))
    out = _finalize_external_df(pd.DataFrame(rows))
    if max_rows and len(out) > max_rows:
        out = _cap_external_per_label(out, max(40, max_rows // 10))
        if len(out) > max_rows:
            out = out.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return out

@st.cache_data(show_spinner=False)
def fetch_counsel_teacher_candidates(max_rows=400):
    load_dataset = _require_datasets_library()
    if load_dataset is None:
        return pd.DataFrame()
    try:
        ds = load_dataset("nbertagnolli/counsel-chat")
    except Exception as e:
        _record_external_fetch_diag('Counsel-Chat', f'load failed -> {e}')
        return pd.DataFrame()

    frames = [pd.DataFrame(ds[key]) for key in ds.keys()]
    df = pd.concat(frames, ignore_index=True)

    text_cols = [col for col in ['questionText', 'answerText', 'question', 'answer', 'text'] if col in df.columns]
    if not text_cols:
        _record_external_fetch_diag('Counsel-Chat', f'schema mismatch -> columns={df.columns.tolist()}')
        return pd.DataFrame()

    rows = []
    for col in text_cols:
        for value in df[col].dropna().astype(str).tolist():
            txt = clean_text(value)
            if len(txt) < 20:
                continue
            rows.append({
                'text': txt,
                'text_hash': build_text_hash(txt),
                'source_dataset': 'counsel_chat',
                'label_origin': 'teacher_candidate',
                'mapping_confidence': 'unlabeled',
                'sample_weight': 0.5,
                'external_label': '',
                'sampler_bucket': 'external_teacher_candidate',
                'activity_type': 'External',
                'community_name': 'counsel_chat',
                'source_commenter_name': 'External Dataset',
                'teacher_data_version': 'external_teacher_candidate_v1',
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.drop_duplicates(subset=['text_hash']).reset_index(drop=True)
    if max_rows and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return out

def append_rows_to_master_dataset(new_rows_df, latest_batch_filename=None, success_prefix="Batch"):
    if new_rows_df is None or new_rows_df.empty:
        return {'added_rows': 0, 'total_rows': None, 'new_key_count': 0}

    master_file = "master_training_data.csv"
    existing_df = pd.read_csv(master_file) if os.path.exists(master_file) else pd.DataFrame()

    for frame in (existing_df, new_rows_df):
        if not frame.empty and 'text_hash' not in frame.columns and 'text' in frame.columns:
            frame['text_hash'] = frame['text'].astype(str).apply(build_text_hash)

    if latest_batch_filename:
        new_rows_df.to_csv(latest_batch_filename, index=False)

    dedupe_key = 'text_hash' if ('text_hash' in new_rows_df.columns or ('text_hash' in existing_df.columns and not existing_df.empty)) else 'text'
    existing_keys = set(existing_df[dedupe_key].dropna().astype(str).tolist()) if (not existing_df.empty and dedupe_key in existing_df.columns) else set()
    new_keys = set(new_rows_df[dedupe_key].dropna().astype(str).tolist()) if (not new_rows_df.empty and dedupe_key in new_rows_df.columns) else set()
    new_key_count = len(new_keys - existing_keys)

    combined_master = pd.concat([existing_df, new_rows_df], ignore_index=True)
    combined_master = combined_master.drop_duplicates(subset=[dedupe_key]).reset_index(drop=True)
    added_rows = len(combined_master) - len(existing_df)

    if added_rows > 0:
        combined_master.to_csv(master_file, index=False)

    st.session_state['last_training_append_message'] = (
        f"✅ {success_prefix}: saved {len(new_rows_df):,} rows. "
        f"Added {max(0, added_rows):,} unique rows to the master dataset. "
        f"Total master size: {len(combined_master):,} rows."
    )
    st.session_state['last_generation_stats'] = {
        'sampled_candidates': int(len(new_rows_df)),
        'teacher_valid_rows': int(len(new_rows_df)),
        'teacher_invalid_or_failed': 0,
        'new_unique_candidate_rows': int(new_key_count),
        'rows_written_to_master': int(max(0, added_rows)),
    }
    return {'added_rows': int(max(0, added_rows)), 'total_rows': int(len(combined_master)), 'new_key_count': int(new_key_count)}

def teacher_label_external_candidates_df(candidate_df, sample_size=300, excluded_hashes=None):
    if candidate_df is None or candidate_df.empty:
        st.warning("No external teacher candidates available.")
        return pd.DataFrame()

    working = candidate_df.copy()
    if 'text_hash' not in working.columns and 'text' in working.columns:
        working['text_hash'] = working['text'].astype(str).apply(build_text_hash)
    working['text'] = working['text'].astype(str).apply(clean_text)
    working = working[working['text'].str.len() >= 12].drop_duplicates(subset=['text_hash']).reset_index(drop=True)

    excluded_hashes = set(excluded_hashes or [])
    if excluded_hashes:
        working = working[~working['text_hash'].isin(excluded_hashes)].reset_index(drop=True)

    if working.empty:
        st.warning("No new unique external candidates remain after deduplication.")
        return pd.DataFrame()

    requested_total = max(1, int(sample_size))
    target_total = min(requested_total, len(working))
    if target_total < requested_total:
        st.info(f"Requested {requested_total:,} external teacher-label rows, but only {len(working):,} unique candidates were available. Using {target_total:,}.")

    sampled_df = working.sample(n=target_total, random_state=42).reset_index(drop=True)
    thresholds = get_label_thresholds()

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.markdown(
        f":material/science: **External teacher-label generation:** Gemini scoring {len(sampled_df):,} external candidates in batches of {PRODUCTION_BATCH_SIZE}..."
    )
    progress_bar.progress(0.0)

    raw_results = process_in_parallel_batched(
        _gemini_distillation_call,
        sampled_df['text'].astype(str).tolist(),
        progress_bar,
        status_text,
        ":material/science: **External teacher-label generation:** Gemini scored {completed:,}/{total:,} items",
        max_workers=10,
        batch_size=PRODUCTION_BATCH_SIZE
    )

    results = []
    for idx, comment in enumerate(sampled_df['text'].astype(str).tolist()):
        raw_text = raw_results.get(idx, "ERROR: Timeout")
        if isinstance(raw_text, str) and raw_text.startswith("ERROR:"):
            continue
        try:
            scores = normalize_emotion_scores(json.loads(raw_text))
            active_labels = get_active_labels_from_scores(scores, thresholds)
            source_row = sampled_df.iloc[idx]
            row_data = {
                'text': comment,
                'labels': ', '.join(active_labels),
                'text_hash': source_row.get('text_hash'),
                'sampler_bucket': 'external_teacher_labeled',
                'activity_type': 'External',
                'community_name': source_row.get('source_dataset', 'external_teacher'),
                'source_commenter_name': 'External Dataset',
                'student_primary_emotion': 'external_teacher',
                'student_primary_score': 0.0,
                'student_negative_sum': 0.0,
                'student_false_positive_signal': 0.0,
                'min_threshold_margin': 0.0,
                'rare_hint_emotions': '',
                'teacher_data_version': 'external_teacher_labeled_v1',
                'teacher_default_threshold': 0.4,
                'source_dataset': source_row.get('source_dataset', 'external_teacher'),
                'label_origin': 'teacher_llm_external',
                'mapping_confidence': 'teacher',
                'sample_weight': float(source_row.get('sample_weight', 0.5)),
                'external_label': source_row.get('external_label', ''),
            }
            for emotion in EMOTION_LABELS:
                row_data[emotion] = scores.get(emotion, 0.0)
            results.append(row_data)
        except Exception:
            continue

    progress_bar.empty()
    status_text.empty()

    result_df = pd.DataFrame(results)
    st.session_state['last_generation_stats'] = {
        'sampled_candidates': int(len(sampled_df)),
        'teacher_valid_rows': int(len(result_df)),
        'teacher_invalid_or_failed': int(max(0, len(sampled_df) - len(result_df))),
    }

    if result_df.empty:
        st.error("External teacher labeling completed, but no valid labeled rows were produced.")
        return pd.DataFrame()

    return result_df


# ==========================================
# --- MLOPS: MODEL EVALUATION TOOL ---
# ==========================================
@st.dialog("📊 Model Evaluation (RoBERTa vs. Gemini)", width="large")
def evaluate_model_performance(df, model_id, sample_size=20):
    st.caption("Dual-track evaluation: (1) Macro F1/Cohen's Kappa emotion agreement against Gemini, and (2) Binary F1 support-triage agreement against a BSF need-aware Gemini reviewer.")

    if sample_size < 100:
        st.warning("⚠️ **Statistical Warning:** Sample sizes under 100 are highly sensitive to outliers. F1 scores may fluctuate wildly. Use N=100+ for reliable drift detection.")

    if df.empty:
        st.warning("No data available to evaluate.")
        return

    eval_pool = df.copy()
    eval_pool['content'] = eval_pool['content'].astype(str).apply(clean_text)
    eval_pool = eval_pool[eval_pool['content'].apply(is_viable_training_text)].drop_duplicates(subset=['content'])
    if eval_pool.empty:
        st.warning("No high-quality evaluation text available.")
        return

    sample_df = eval_pool.sample(min(sample_size, len(eval_pool)), random_state=42).reset_index(drop=True)
    comments_to_process = sample_df['content'].astype(str).tolist()

    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.markdown(f":material/analytics: **Emotion agreement pass:** Gemini evaluating {len(sample_df)} items...")
    progress_bar.progress(0.0)
    raw_emotion_results = process_in_parallel(
        _gemini_distillation_call,
        comments_to_process,
        progress_bar,
        status_text,
        ":material/analytics: **Emotion agreement pass:** {completed}/{total}"
    )

    status_text.markdown(f":material/volunteer_activism: **BSF support-review pass:** Gemini reviewing {len(sample_df)} items...")
    progress_bar.progress(0.0)
    raw_support_results = process_in_parallel(
        _gemini_support_eval_cached,
        comments_to_process,
        progress_bar,
        status_text,
        ":material/volunteer_activism: **BSF support-review pass:** {completed}/{total}"
    )

    student_scores_df = analyze_emotions_cached(pd.DataFrame({'content': comments_to_process}), model_id=model_id).reset_index(drop=True)
    student_thresholds = get_label_thresholds()
    teacher_thresholds = student_thresholds.copy()

    y_true_all, y_pred_all = [], []
    y_true_support, y_pred_support = [], []
    per_label_truth = {emotion: [] for emotion in EMOTION_LABELS}
    per_label_pred = {emotion: [] for emotion in EMOTION_LABELS}
    eval_results = []

    for idx, comment in enumerate(comments_to_process):
        teacher_labels = "Teacher error"
        student_labels = "Unavailable"
        biggest_disagreement = "N/A"
        student_scores = None

        emotion_raw_text = raw_emotion_results.get(idx, "ERROR: Timeout")
        if not (isinstance(emotion_raw_text, str) and emotion_raw_text.startswith("ERROR:")):
            try:
                teacher_scores = normalize_emotion_scores(json.loads(emotion_raw_text))
                student_scores = normalize_emotion_scores(student_scores_df.iloc[idx].to_dict())

                for emotion in EMOTION_LABELS:
                    teacher_bin = 1 if teacher_scores.get(emotion, 0.0) >= teacher_thresholds.get(emotion, 0.4) else 0
                    student_bin = 1 if student_scores.get(emotion, 0.0) >= student_thresholds.get(emotion, 0.4) else 0
                    y_true_all.append(teacher_bin)
                    y_pred_all.append(student_bin)
                    per_label_truth[emotion].append(teacher_bin)
                    per_label_pred[emotion].append(student_bin)

                deltas = {emotion: round(student_scores.get(emotion, 0.0) - teacher_scores.get(emotion, 0.0), 2) for emotion in EMOTION_LABELS}
                max_diff_emo = max(deltas, key=lambda label: abs(deltas[label]))
                teacher_labels = ', '.join(get_active_labels_from_scores(teacher_scores, teacher_thresholds))
                student_labels = ', '.join(get_active_labels_from_scores(student_scores, student_thresholds))
                biggest_disagreement = f"{max_diff_emo.title()} ({deltas[max_diff_emo]:+.2f})"
            except Exception:
                pass

        support_raw_text = raw_support_results.get(idx, "ERROR: Timeout")
        teacher_support = None if (isinstance(support_raw_text, str) and support_raw_text.startswith("ERROR:")) else parse_gemini_boolean_verdict(support_raw_text)
        student_support = compute_severe_candidate_details(student_scores, comment)['candidate'] if student_scores is not None else None

        if teacher_support is not None and student_support is not None:
            y_true_support.append(int(teacher_support))
            y_pred_support.append(int(student_support))

        eval_results.append({
            'Text Preview': comment[:80] + ('...' if len(comment) > 80 else ''),
            'Teacher (Gemini)': teacher_labels,
            'Student (RoBERTa)': student_labels,
            'BSF Need Verdict': (
                f'Teacher={teacher_support} / Student={student_support}'
                if teacher_support is not None and student_support is not None else 'Unavailable'
            ),
            'Biggest Disagreement': biggest_disagreement
        })

    progress_bar.empty()
    status_text.empty()

    if not y_true_all:
        st.error("Evaluation failed. Could not retrieve enough comparable emotion scores.")
        return

    f1 = f1_score(y_true_all, y_pred_all, average='macro')
    kappa = cohen_kappa_score(y_true_all, y_pred_all)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])

    n_samples = len(y_true_all)
    ci_margin = 1.96 * math.sqrt((f1 * (1 - f1)) / n_samples) if n_samples > 0 else 0

    per_label_rows = []
    for emotion in EMOTION_LABELS:
        truth = per_label_truth[emotion]
        pred = per_label_pred[emotion]
        if not truth:
            continue
        precision, recall, label_f1, _ = precision_recall_fscore_support(truth, pred, average='binary', zero_division=0)
        per_label_rows.append({
            'Emotion': emotion.title(),
            'Threshold': student_thresholds.get(emotion, 0.4),
            'Support': int(sum(truth)),
            'Predicted': int(sum(pred)),
            'Precision': round(float(precision), 3),
            'Recall': round(float(recall), 3),
            'F1': round(float(label_f1), 3),
        })

    label_metrics_df = pd.DataFrame(per_label_rows).sort_values(['F1', 'Support'], ascending=[True, False]) if per_label_rows else pd.DataFrame()
    watchlist_df = label_metrics_df[label_metrics_df['Emotion'].isin(SEVERE_WATCHLIST_LABELS)].copy() if not label_metrics_df.empty else pd.DataFrame()
    watchlist_macro_f1 = round(float(watchlist_df['F1'].mean()), 3) if not watchlist_df.empty else 0.0

    if y_true_support:
        support_precision, support_recall, support_f1, _ = precision_recall_fscore_support(y_true_support, y_pred_support, average='binary', zero_division=0)
        support_cm = confusion_matrix(y_true_support, y_pred_support, labels=[0, 1])
        tn, fp, fn, tp = support_cm.ravel()
        benign_fp_rate = float(fp / max(tn + fp, 1))
    else:
        support_precision, support_recall, support_f1 = 0.0, 0.0, 0.0
        benign_fp_rate = 0.0
        support_cm = np.array([[0, 0], [0, 0]])
        tn = fp = fn = tp = 0

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric('Emotion Macro F1', f"{f1:.2f} ± {ci_margin:.2f}", help='Flattened macro F1 across all emotion decisions with 95% Confidence Interval.')
    with col2:
        st.metric('Emotion Kappa', f"{kappa:.2f}", help='Cohen\'s Kappa agreement between the student and teacher emotion labels.')
    with col3:
        st.metric('Severe Watchlist F1', f"{watchlist_macro_f1:.2f}", help='Average F1 on severe or intervention-oriented labels.')
    with col4:
        st.metric('BSF Support F1', f"{support_f1:.2f}", help='Agreement with the BSF need-aware Gemini reviewer on whether an item should be surfaced for support review.')
    with col5:
        st.metric('Benign FP Rate', f"{benign_fp_rate:.2f}", help='Fraction of teacher-negative items that the model still sends to support review.')

    st.caption(
        f"BSF support precision: {support_precision:.2f} | BSF support recall: {support_recall:.2f} | "
        f"TP={tp}, FP={fp}, FN={fn}, TN={tn}"
    )

    st.markdown("#### Global Emotion Confusion Matrix (All Labels)")
    z = cm[::-1]
    x = ['Student Predicted: 0', 'Student Predicted: 1']
    y = ['Teacher Actual: 1', 'Teacher Actual: 0']
    fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
    fig.update_layout(height=400, margin=dict(t=50, l=100))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### BSF Support-Triage Confusion Matrix")
    support_z = support_cm[::-1]
    support_x = ['Student Escalate: 0', 'Student Escalate: 1']
    support_y = ['Teacher Escalate: 1', 'Teacher Escalate: 0']
    support_fig = ff.create_annotated_heatmap(support_z, x=support_x, y=support_y, colorscale='Oranges', showscale=True)
    support_fig.update_layout(height=400, margin=dict(t=50, l=120))
    st.plotly_chart(support_fig, use_container_width=True)

    if not watchlist_df.empty:
        st.markdown('#### Severe / High-Priority Label Watchlist')
        st.dataframe(watchlist_df.sort_values(['F1', 'Support'], ascending=[True, False]), use_container_width=True)

    if not label_metrics_df.empty:
        st.markdown('#### Per-Label Metrics')
        st.dataframe(label_metrics_df, use_container_width=True)

    eval_df = pd.DataFrame(eval_results)
    st.markdown('#### Example Disagreements')
    st.dataframe(eval_df, use_container_width=True)

# --- HUGGING FACE AUTO-DISCOVERY ---
@st.cache_data(ttl=300) # Cache for 5 minutes
def fetch_custom_models(hf_username):
    base_model = "SamLowe/roberta-base-go_emotions"
    if not hf_username:
        return [base_model]
    try:
        api = HfApi()
        seen = set()
        model_ids = []
        for search_term in ["BSF-Custom-Emotions", "BSF-LoRA-Emotions"]:
            for model in api.list_models(author=hf_username, search=search_term):
                if model.id not in seen:
                    seen.add(model.id)
                    model_ids.append(model.id)
        model_ids.sort(reverse=True)  # Newest first by name stamp
        return [base_model] + model_ids
    except Exception:
        return [base_model]

@st.cache_data(ttl=300, show_spinner=False)
def fetch_model_repo_thresholds(model_id):
    base_model = "SamLowe/roberta-base-go_emotions"
    if not model_id or model_id == base_model:
        return None
    try:
        threshold_path = hf_hub_download(repo_id=model_id, filename='recommended_thresholds.json', repo_type='model')
        with open(threshold_path, 'r') as f:
            return validate_threshold_map(json.load(f))
    except Exception:
        return None


def set_active_threshold_state(thresholds, source_mode, source_label, model_id):
    st.session_state['active_label_thresholds'] = validate_threshold_map(thresholds)
    st.session_state['threshold_source_mode'] = source_mode
    st.session_state['threshold_source_label'] = source_label
    st.session_state['threshold_source_model_id'] = model_id


def sync_thresholds_for_active_model(model_id, force_refresh=False):
    base_model = "SamLowe/roberta-base-go_emotions"
    current_model = st.session_state.get('threshold_source_model_id')
    current_mode = st.session_state.get('threshold_source_mode')
    has_session_thresholds = isinstance(st.session_state.get('active_label_thresholds'), dict) and bool(st.session_state.get('active_label_thresholds'))

    if not force_refresh and current_model == model_id and current_mode in {'base', 'model_repo', 'manual_upload', 'model_repo_missing'} and has_session_thresholds:
        return

    if model_id == base_model:
        set_active_threshold_state(
            DEFAULT_LABEL_THRESHOLDS,
            'base',
            'Base-model calibrated per-emotion thresholds',
            model_id,
        )
        return

    repo_thresholds = fetch_model_repo_thresholds(model_id)
    if repo_thresholds:
        set_active_threshold_state(
            repo_thresholds,
            'model_repo',
            'Auto-loaded calibrated thresholds from the active model repo',
            model_id,
        )
    else:
        set_active_threshold_state(
            DEFAULT_LABEL_THRESHOLDS,
            'model_repo_missing',
            'No model-specific threshold file found; using base-model calibrated thresholds',
            model_id,
        )


def render_threshold_source_badge():
    source_label = st.session_state.get('threshold_source_label', 'Base-model calibrated per-emotion thresholds')
    source_mode = st.session_state.get('threshold_source_mode', 'base')
    badge_styles = {
        'base': ('#eef6ff', '#1f6feb'),
        'model_repo': ('#eafaf1', '#1f883d'),
        'manual_upload': ('#fff8e6', '#b26a00'),
        'model_repo_missing': ('#fff5f5', '#cf222e'),
    }
    bg, border = badge_styles.get(source_mode, ('#f4f4f4', '#666'))
    st.markdown(
        f"""
        <div style="margin: 0.35rem 0 0.75rem 0; padding: 0.65rem 0.8rem; border-radius: 0.55rem; background: {bg}; border-left: 4px solid {border};">
            <div style="font-size: 0.78rem; font-weight: 700; color: #333; margin-bottom: 0.15rem;">🎯 Current threshold source</div>
            <div style="font-size: 0.83rem; color: #333; line-height: 1.35;">{source_label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def reset_ab_thresholds_to_defaults():
    """Reset A/B scam confidence threshold experiment controls to baseline defaults."""
    st.session_state['ab_block_threshold'] = 0.85
    st.session_state['ab_medium_threshold'] = 0.60

# --- Main Dashboard Function ---
def run_dashboard():
    if 'analysis_run' not in st.session_state:
        st.session_state.analysis_run = False
        st.session_state.results = {}
    if "sanity_log" not in st.session_state:
        st.session_state.sanity_log =[]
    if 'threshold_source_mode' not in st.session_state:
        st.session_state['threshold_source_mode'] = 'base'
    if 'threshold_source_label' not in st.session_state:
        st.session_state['threshold_source_label'] = 'Base-model calibrated per-emotion thresholds'
    if 'threshold_source_model_id' not in st.session_state:
        st.session_state['threshold_source_model_id'] = 'SamLowe/roberta-base-go_emotions'

    # Safety defaults for recently added scam-review controls
    st.session_state.setdefault('ab_block_threshold', 0.85)
    st.session_state.setdefault('ab_medium_threshold', 0.60)
    st.session_state.setdefault('scam_feedback_log', [])
    st.session_state.setdefault('fn_rate_history', [])
    st.session_state.setdefault('audit_queue', [])
    st.session_state.setdefault('scam_include_trusted_users', False)

    try:
        logo_base64 = get_image_as_base64("logo.png")
        st.markdown(f"""
        <style>
            .header-container {{ display: flex; align-items: center; justify-content: flex-start; background-color: #F8F9FA; padding: 1rem 1.5rem; border-radius: 10px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08); margin-bottom: 2rem; }}
            .header-logo {{ width: 120px; margin-right: 1.5rem; }}
            .header-text {{ display: flex; flex-direction: column; justify-content: center; }}
            .header-text h1 {{ font-weight: 700; color: #00387b; font-size: 2.2rem; margin: 0; line-height: 1.2; }}
            .header-text p {{ font-size: 1.1rem; color: #d32d27; margin: 0; padding-top: 0.25rem; }}
            [data-testid="stAppViewContainer"] > .main {{ background-image: url("data:image/png;base64,{logo_base64}"); background-size: 300px; background-position: top right; background-repeat: no-repeat; background-attachment: fixed; }}
            [data-testid="stAppViewContainer"] > .main::before {{ content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(255, 255, 255, 0.90); z-index: -1; }}
            .metric-definition {{ background-color: #f0f8ff; padding: 0.8rem; border-radius: 0.5rem; margin: 0.5rem 0; border-left: 4px solid #1f77b4; }}
            .metric-definition h5 {{ margin-bottom: 5px; font-size: 1rem; color: #00387b; }}
            .metric-definition p {{ font-size: 0.85rem; margin-bottom: 5px; color: #333; line-height: 1.4; }}
            .metric-definition ul {{ font-size: 0.85rem; margin-bottom: 5px; padding-left: 20px; color: #333; }}
            .metric-definition .logic {{ font-family: monospace; font-size: 0.8rem; color: #555; background: #e9ecef; padding: 2px 5px; border-radius: 3px; }}
            .st-emotion-cache-1v0mbdj > img {{ border-radius: 8px; }}
        </style>
        <div class="header-container"><img src="data:image/png;base64,{logo_base64}" class="header-logo"><div class="header-text"><h1>Community Wellness Dashboard</h1><p>Emotional Health Insights from Blue Star Families Forum Discussions</p></div></div>
        """, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Logo file not found. Please ensure 'logo.png' is in the app directory.")

    with st.sidebar.expander("⚙️ Analysis Controls", expanded=False):
        short_sha, sha_source = _get_build_sha()
        st.success(
            f"🟢 **Build:** `{short_sha}` ({sha_source})  \n"
            f"calibration-v5 + fraud-3tier"
        )
        if st.button("🔄 Clear caches & rerun", help="Forces fresh emotion analysis + scam scan with the latest calibration code", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.session_state.pop("results", None)
            st.session_state["analysis_run"] = False
            st.rerun()
    with st.sidebar.expander("🔍 Diagnostics", expanded=False):
        st.caption(f"**Active model:** `{st.session_state.get('active_model_id', 'SamLowe/roberta-base-go_emotions')}`")
        st.caption(f"**Threshold source:** `{st.session_state.get('threshold_source_mode', 'base')}` — {st.session_state.get('threshold_source_label', 'default')}")
        last_flagged = st.session_state.get('last_initial_flagged_count')
        if last_flagged is not None:
            st.caption(f"**Last run · initial Tier-1 severe candidates:** `{last_flagged}`")
        last_scam_t1 = st.session_state.get('last_scam_tier1_count')
        if last_scam_t1 is not None:
            st.caption(f"**Last run · Tier-1 scam candidates:** `{last_scam_t1}`")
        last_dbg = st.session_state.get('last_severe_debug')
        if last_dbg is not None:
            st.caption("**Severe detection trace (last run):**")
            st.json(last_dbg, expanded=False)
        clf_diag = st.session_state.get('emotion_classifier_diag')
        if clf_diag is not None:
            st.caption("**Emotion classifier load:**")
            st.json(clf_diag, expanded=False)
        try:
            _wcache = _load_weekly_verified_cache()
            _mirror_keys = set((st.session_state.get('weekly_verified_cache_mirror') or {}).keys())
            _file_only = sorted(k for k in _wcache.keys() if k not in _mirror_keys)
            _session_only = sorted(_mirror_keys - set(_wcache.keys()) | _mirror_keys & set(_wcache.keys()))
            st.caption(f"**Weekly verified cache:** `{len(_wcache)}` week(s) stored (session mirror: `{len(_mirror_keys)}`)")
            if _wcache:
                st.json({k: {"severe": v.get("severe_count"), "scam": v.get("scam_count"), "comments": v.get("comment_count"), "src": ("session" if k in _mirror_keys else "file")} for k, v in sorted(_wcache.items())}, expanded=False)
            _save_err = st.session_state.get('weekly_cache_save_error')
            _load_err = st.session_state.get('weekly_cache_load_error')
            if _save_err:
                st.caption(f"⚠️ Last disk save error: `{_save_err}`")
            if _load_err:
                st.caption(f"⚠️ Last disk load error: `{_load_err}`")
        except Exception:
            pass
        clf_baseline = st.session_state.get('emotion_baseline_fallback_count')
        if clf_baseline:
            st.error(
                f"⚠️ RoBERTa scoring fell back to baseline (neutral=0.85) for "
                f"{clf_baseline} interaction(s). Severe detection is unreliable. "
                f"Last error: {st.session_state.get('emotion_baseline_last_error', 'unknown')}"
            )
    current_year = datetime.now().year
    years =[str(y) for y in range(2024, current_year + 2)]
    selected_year = st.sidebar.selectbox("📅 Select Year", years, index=years.index(str(current_year)))
    
    week_choices = get_week_choices(int(selected_year))
    current_week_num = datetime.now().isocalendar().week
    
    try:
        current_week_start = datetime.strptime(f'{selected_year}-W{current_week_num:02d}-1', "%G-W%V-%u")
        current_month = current_week_start.strftime("%b")
        current_week_choice = f"Week {current_week_num} ({current_month})"
    except ValueError:
        current_week_choice = week_choices[-1] if week_choices else ""

    selected_week_str = st.sidebar.selectbox("📅 Select Week", week_choices, index=week_choices.index(current_week_choice) if current_week_choice in week_choices else 0)

    analyze_button = st.sidebar.button("Analyze", type="primary", icon=":material/search:")

    st.sidebar.markdown("---")
    with st.sidebar.expander("📖 Metric Definitions", expanded=False):
        st.markdown("""
            <div class="metric-definition"><h5>Health Score</h5><p>A weekly population-level emotional-balance indicator on a 0–10 scale. It does <b>not</b> measure crisis risk directly; it summarizes how much of the week's total emotion mass falls into negative labels versus all labels combined.</p><p><span class="logic">10 * (1 - (∑ negative emotion intensity / ∑ all emotion intensity))</span></p></div>
            <div class="metric-definition"><h5>Emotion Cutoffs</h5><p>Each emotion uses its own calibrated decision threshold because labels are not equally reliable or equally severe. The app first tries to load <code>recommended_thresholds.json</code> from the active model repo; if none is found, it falls back to conservative base thresholds, especially for rare high-risk labels.</p><p><span class="logic">emotion_i is active when score_i ≥ threshold_i</span></p></div>
            <div class="metric-definition"><h5>Severe Interactions</h5><p>The severe-review path is intentionally <b>High Recall First, High Precision Second</b>. A post becomes a review candidate when the RoBERTa model shows rare/severe review-band evidence, a severe-negative cluster, or a BSF support-need signal. Gemini then acts as a <b>precision filter</b> utilizing <b>Chain-of-Thought (CoT) reasoning</b> to confirm whether the post reflects a real member need, true concern, or intervention-worthy distress before escalation.</p><p><b>Display policy:</b> the dashboard's Severe list and bar chart show <i>only</i> items where Gemini returned <code>True</code> (<code>verdict_source == 'gemini_true'</code>). Items kept by the internal recall-override (Gemini False/ambiguous but strong hardship evidence) are still recorded for email alerts and audit logs but are hidden from the visible dashboard to avoid surfacing low-precision cases.</p><p><span class="logic">candidate = review-band trigger OR severe cluster OR support-need signal; displayed only if Gemini verdict = True</span></p></div>
            <div class="metric-definition"><h5>Mild-Moderate Interactions</h5><p>Interactions with meaningful negative load that do not meet the severe-review criteria. These items are kept for trend monitoring and early-signal analysis so the dashboard can surface emerging strain without over-escalating routine frustration or low-stakes distress.</p><p><span class="logic">mild floor ≤ negative load &lt; severe candidate criteria</span></p></div>
        """, unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown("## :material/memory: System Architecture")
    st.sidebar.markdown("""
        <div class="metric-definition" style="background-color: #e8f4f8; border-left-color: #007bff;">
            <h5>1. Student Model (Primary Inference Layer)</h5>
            <p>A RoBERTa-based multilabel emotion classifier runs batched inference (<code>batch_size=32</code>) utilizing <b>Hardware Acceleration (CUDA/MPS)</b> and <b>Sliding Window Chunking</b> (300-word spans with 50-word overlap) to prevent truncation loss on long posts. It emits per-label probabilities. Those probabilities are converted into label decisions and review signals using calibrated per-emotion thresholds rather than one shared global cutoff.</p>
            <h5>2. Gemini Review Layer (BSF Need-Aware Precision Filter)</h5>
            <p>Gemini Flash reviews only selected candidates using <b>Chain-of-Thought (CoT) reasoning</b>. Its role is not generic sentiment validation; it forces the LLM to think step-by-step to decide whether a post indicates a <b>real member need, actionable concern, or intervention-worthy distress</b> in Blue Star Families context. The same layer also verifies higher-risk scam/fraud candidates after deterministic heuristic pre-filtering.</p>
            <h5>3. Teacher-Student Data Improvement Loop</h5>
            <p>Gemini 3.1 Pro is used for teacher-generated soft labels and weak supervision. It uses <b>Shannon Entropy Uncertainty Sampling</b> to mathematically identify the most uncertain predictions for active learning on rare-label cases, threshold-boundary items, and hard negatives. This creates a higher-value training set for iterative fine-tuning and LoRA experiments without depending entirely on manual labeling.</p>
        </div>
    """, unsafe_allow_html=True)

    # --- MLOPS EVALUATION SECTION ---
    st.sidebar.markdown("---")
    st.sidebar.markdown("## :material/analytics: MLOps Evaluation")
    st.sidebar.markdown("""
        <div class="metric-definition" style="background-color: #f8f9fa; border-left-color: #6c757d;">
            <p>This evaluation separates <b>two different failure modes</b>: <b>(1) emotion-model drift</b>, measured with Macro F1 / Cohen's Kappa against Gemini emotion labels, and <b>(2) BSF support-triage drift</b>, measured with Gemini 3.1 Pro on whether the system would surface the same posts for support review. It explicitly tracks support precision, support recall, and benign false-positive rate so you can distinguish general label-quality problems from operational escalation problems on real BSF content. It now includes <b>95% Confidence Intervals</b> to ensure statistical significance before triggering drift alerts.</p>
        </div>
    """, unsafe_allow_html=True)
    
    eval_sample_size = st.sidebar.slider("Eval Sample Size", min_value=20, max_value=200, value=100, step=20)
    if st.sidebar.button("Run Model Evaluation", icon=":material/bar_chart:"):
        if st.session_state.analysis_run and 'df' in st.session_state.results:
            active_model_id = st.session_state.get('active_model_id', "SamLowe/roberta-base-go_emotions")
            with st.spinner("Evaluating model against Gemini 3.1 Pro..."):
                evaluate_model_performance(st.session_state.results['df'], model_id=active_model_id, sample_size=eval_sample_size)
        else:
            st.sidebar.warning("Please run the Weekly Analysis first!")

    # --- A/B CONFIDENCE THRESHOLD EXPERIMENT ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("🧪 A/B Threshold Experiment", expanded=False):
        st.markdown("""
            <div class="metric-definition" style="background-color: #fef6e4; border-left-color: #f4a261;">
                <h5>A/B Confidence Threshold Testing</h5>
                <p>Adjust the scam-detection confidence thresholds to compare flag volume vs. precision. 
                Lower thresholds increase recall (catch more scams) but raise false positives. 
                Higher thresholds improve precision but risk missing content. These values now drive both escalation policy and severity tiering.</p>
            </div>
        """, unsafe_allow_html=True)

        ab_block_threshold = st.slider(
            "Scam BLOCK threshold (auto-escalate)",
            min_value=0.70, max_value=0.99, value=float(st.session_state.get("ab_block_threshold", 0.85)), step=0.01,
            help="Scam flags with confidence ≥ this value are immediately escalated.",
            key="ab_block_threshold"
        )
        ab_medium_threshold = st.slider(
            "Medium severity floor",
            min_value=0.50, max_value=0.85, value=float(st.session_state.get("ab_medium_threshold", 0.60)), step=0.01,
            help="Flags with confidence ≥ this value but below BLOCK threshold enter MEDIUM queue.",
            key="ab_medium_threshold"
        )

        reset_col, current_col = st.columns([1, 1])
        with reset_col:
            st.button(
                "↺ Reset A/B defaults",
                key="ab_reset_defaults_btn",
                on_click=reset_ab_thresholds_to_defaults,
                use_container_width=True,
                help="Restores BLOCK=0.85 and MEDIUM floor=0.60"
            )
        with current_col:
            st.caption(
                f"Current: BLOCK={st.session_state.get('ab_block_threshold', 0.85):.2f}, "
                f"MEDIUM={st.session_state.get('ab_medium_threshold', 0.60):.2f}"
            )

        fb_log = st.session_state.get("scam_feedback_log", [])
        if fb_log:
            total    = len(fb_log)
            fp_count = sum(1 for f in fb_log if f["verdict"] == "FALSE_POSITIVE")
            tp_count = sum(1 for f in fb_log if f["verdict"] == "CONFIRMED_SCAM")
            st.markdown("**Current session stats (from HITL feedback):**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Confirmed Scams",  tp_count)
            c2.metric("False Positives",  fp_count)
            c3.metric("Precision",
                      f"{tp_count / max(tp_count + fp_count, 1):.0%}",
                      help="TP / (TP + FP) from staff verdicts")
            if fp_count > 0:
                st.warning(f"⚠️ {fp_count} false positive(s) detected. Consider raising the BLOCK threshold.")
            else:
                st.success("✅ No false positives marked yet.")
        else:
            st.caption("Run an analysis and mark feedback to see live precision metrics here.")

    # --- CONTINUOUS LEARNING & COLAB INTEGRATION ---
    st.sidebar.markdown("---")
    with st.sidebar.expander("🚀 Continuous Learning Pipeline", expanded=False):
        st.markdown("""
            <div class="metric-definition" style="background-color: #e6f2ff; border-left-color: #005cbf;">
                <h5>Active Learning Strategy</h5>
                <p>This section implements a production, label-aware data engine that intentionally oversamples the examples most useful for model improvement: <b>rare-label cases (30%)</b>, <b>Shannon Entropy uncertain cases (25%)</b> (mathematically identifying where the model is most confused), <b>false-positive suspects (20%)</b>, <b>neutral baseline controls (15%)</b>, and <b>random baseline coverage (10%)</b>. The purpose is to improve rare-label coverage and support-triage precision without simply adding more easy majority-class examples.</p>
                <hr style="margin: 8px 0; border-top: 1px solid #b3d7ff;">
                <p><b>1. Generate:</b> Teacher models score recent in-domain data and selected external candidates.<br>
                <b>2. Deduplicate:</b> Hash-based filtering prevents repeated training examples from re-entering the master dataset.<br>
                <b>3. Fine-Tune:</b> Newly accumulated batches feed GPU fine-tuning / LoRA experiments.<br>
                <b>4. Deploy:</b> The active Hugging Face model and its thresholds can be hot-swapped below.</p>
            </div>
        """, unsafe_allow_html=True)
        
        lookback_weeks = st.slider( "Historical Data Range (Weeks)", min_value=1, max_value=100, value=12, help="How many weeks back to pull data from for training generation." )
        sample_size = st.slider("Target Sample Size", min_value=100, max_value=2000, value=1000, step=100, help="Production-safe collector. For major retraining rounds, use 1500-2000 when enough unique weekly data is available.")
        
        if st.button("1️⃣ Generate & Append Data", icon=":material/database:", use_container_width=True):
            if not st.session_state.analysis_run:
                st.warning("Please run the Weekly Analysis first!")
            else:
                active_model_id = st.session_state.get('active_model_id', "SamLowe/roberta-base-go_emotions")
                with st.spinner(f"Fetching {lookback_weeks} weeks of data and generating a production-grade label-aware training batch with the Gemini 3.1 Pro teacher..."):
                    all_dfs = []
                    all_em_dfs = []
                    year = int(selected_year)
                    week_num = extract_week_number(selected_week_str)
                    
                    try: 
                        start_date_of_selected_week, _ = get_week_dates(year, week_num)
                        for offset in range(lookback_weeks):
                            target_date = start_date_of_selected_week - timedelta(weeks=offset)
                            y, w, _ = target_date.isocalendar()
                            
                            temp_df = load_data_for_week(w, y, show_progress=False)
                            if not temp_df.empty:
                                temp_em = analyze_emotions_cached(temp_df, model_id=active_model_id)
                                all_dfs.append(temp_df)
                                all_em_dfs.append(temp_em)
                    except ValueError:
                        pass
                    
                    if all_dfs:
                        combined_df = pd.concat(all_dfs, ignore_index=True)
                        combined_df_em = pd.concat(all_em_dfs, ignore_index=True)
                        
                        master_file = "master_training_data.csv"
                        if os.path.exists(master_file):
                            existing_df = pd.read_csv(master_file)
                        else:
                            existing_df = pd.DataFrame()

                        if not existing_df.empty and 'text_hash' not in existing_df.columns and 'text' in existing_df.columns:
                            existing_df['text_hash'] = existing_df['text'].astype(str).apply(build_text_hash)
                        existing_hashes = set(existing_df['text_hash'].dropna().astype(str).tolist()) if ('text_hash' in existing_df.columns and not existing_df.empty) else set()

                        training_df = generate_distillation_data(
                            combined_df,
                            combined_df_em,
                            sample_size=sample_size,
                            excluded_hashes=existing_hashes
                        )
                        if training_df is not None:
                            latest_batch_file = "latest_generated_training_batch.csv"
                            training_df.to_csv(latest_batch_file, index=False)
                            st.session_state['last_generated_batch_rows'] = int(len(training_df))

                            for frame in (existing_df, training_df):
                                if not frame.empty and 'text_hash' not in frame.columns and 'text' in frame.columns:
                                    frame['text_hash'] = frame['text'].astype(str).apply(build_text_hash)

                            dedupe_key = 'text_hash' if ('text_hash' in training_df.columns or ('text_hash' in existing_df.columns and not existing_df.empty)) else 'text'
                            existing_keys = set(existing_df[dedupe_key].dropna().astype(str).tolist()) if (not existing_df.empty and dedupe_key in existing_df.columns) else set()
                            training_keys = set(training_df[dedupe_key].dropna().astype(str).tolist()) if (not training_df.empty and dedupe_key in training_df.columns) else set()
                            new_key_count = len(training_keys - existing_keys)

                            combined_master = pd.concat([existing_df, training_df], ignore_index=True)
                            combined_master = combined_master.drop_duplicates(subset=[dedupe_key]).reset_index(drop=True)
                            added_rows = len(combined_master) - len(existing_df)

                            stats = st.session_state.get('last_generation_stats', {})
                            stats['new_unique_candidate_rows'] = int(new_key_count)
                            stats['rows_written_to_master'] = int(max(0, added_rows))
                            st.session_state['last_generation_stats'] = stats

                            if added_rows <= 0:
                                st.warning("No new unique rows were added to the master dataset. The teacher-labeled batch was still saved separately as latest_generated_training_batch.csv.")
                            else:
                                combined_master.to_csv(master_file, index=False)

                            bucket_mix = training_df['sampler_bucket'].value_counts().to_dict() if ('sampler_bucket' in training_df.columns and not training_df.empty) else {}
                            st.session_state['last_training_append_message'] = (
                                f"✅ Teacher batch saved with {len(training_df):,} rows. "
                                f"Added {max(0, added_rows):,} unique rows to the master dataset. "
                                f"Total master size: {len(combined_master):,} rows."
                            )
                            st.session_state['last_training_bucket_mix'] = bucket_mix
                            st.rerun()
                    else:
                        st.error(f"No data found for the last {lookback_weeks} weeks.")


        st.markdown("### 🌐 2. External Data Augmentation")
        st.caption("External data is treated as a controlled supplement, not a blind bulk append. GoEmotions is used mainly for benchmarking or a capped rare-label supplement, mapped datasets are previewed before append because they are noisier than in-domain labels, and unlabeled counseling text is teacher-labeled with Gemini before it can enter the master dataset.")

        datasets_available = has_datasets_library()
        if not datasets_available:
            st.warning("⚠️ External augmentation fetch requires the optional `datasets` package. Install dependencies and restart to enable these fetch controls.")

        if 'external_preview_ready' not in st.session_state:
            st.session_state['external_preview_ready'] = False
        if 'external_preview_summary' not in st.session_state:
            st.session_state['external_preview_summary'] = {}

        ext_col1, ext_col2 = st.columns(2)
        with ext_col1:
            go_rows = st.slider("GoEmotions rows to fetch (benchmark / rare support)", min_value=0, max_value=3000, value=800, step=100, help="Fetched for benchmarking and optional rare-label supplement only. Not bulk-appended by default.")
            ed_rows = st.slider("EmpatheticDialogues mapped rows", min_value=0, max_value=1500, value=300, step=50, help="Conservative mapped supplement for rarer emotional language.")
            include_go_rare = st.checkbox("Include a capped GoEmotions rare-label supplement in preview", value=False, help="Recommended only as a small weighted supplement because the base model already trained on GoEmotions.")
        with ext_col2:
            isear_rows = st.slider("ISEAR mapped rows", min_value=0, max_value=1500, value=300, step=50, help="Mapped supplement especially useful for fear/remorse-adjacent language.")
            counsel_rows = st.slider("Counsel-Chat teacher candidates", min_value=0, max_value=2000, value=300, step=50, help="Unlabeled candidate text to teacher-label with Gemini before append.")
            include_mapped_rows = st.checkbox("Include mapped supplemental rows in preview", value=False, help="Preview first. These rows are useful, but noisier than in-domain or exact-label data.")
        go_cap_per_label = st.slider("GoEmotions rare cap per label", min_value=20, max_value=120, value=60, step=10, help="Applies only if the rare-label GoEmotions supplement is enabled.")

        if st.button("🌐 Fetch External Data & Build Preview", icon=":material/cloud_download:", use_container_width=True, disabled=not datasets_available):
            _clear_external_fetch_diagnostics()
            with st.spinner("Fetching and normalizing external augmentation datasets..."):
                go_df = fetch_goemotions_exact_rows(go_rows) if go_rows > 0 else pd.DataFrame()
                ed_df = fetch_ed_mapped_rows(ed_rows) if ed_rows > 0 else pd.DataFrame()
                isear_df = fetch_isear_mapped_rows(isear_rows) if isear_rows > 0 else pd.DataFrame()
                mapped_df = _finalize_external_df(pd.concat([ed_df, isear_df], ignore_index=True)) if (not ed_df.empty or not isear_df.empty) else pd.DataFrame()
                counsel_df = fetch_counsel_teacher_candidates(counsel_rows) if counsel_rows > 0 else pd.DataFrame()

                if not go_df.empty:
                    go_df.to_csv("exact_append_ready.csv", index=False)
                if not mapped_df.empty:
                    mapped_df.to_csv("mapped_append_ready.csv", index=False)
                if not counsel_df.empty:
                    counsel_df.to_csv("external_teacher_candidates.csv", index=False)

                preview_df = external_append_preview_df(
                    include_go_rare=include_go_rare,
                    include_mapped=include_mapped_rows,
                    go_cap_per_label=go_cap_per_label
                )
                if not preview_df.empty:
                    preview_df.to_csv("combined_external_append_ready.csv", index=False)

                st.session_state['external_preview_ready'] = True
                st.session_state['external_preview_summary'] = build_external_preview_summary(preview_df)
                st.session_state['last_external_fetch_message'] = (
                    f"Fetched external data -> GoEmotions raw: {len(go_df):,}, "
                    f"mapped raw: {len(mapped_df):,}, teacher candidates: {len(counsel_df):,}, "
                    f"preview append-ready rows: {len(preview_df):,}."
                )
                st.rerun()

        if st.session_state.get('last_external_fetch_message'):
            st.info(st.session_state['last_external_fetch_message'])
            summary = st.session_state.get('external_preview_summary', {})
            if summary:
                st.caption(f"Preview append-ready rows: {summary.get('preview_rows', 0):,}")
                if summary.get('source_counts'):
                    st.caption(f"Preview source mix: {summary.get('source_counts')}")
                if summary.get('weight_counts'):
                    st.caption(f"Preview sample weights: {summary.get('weight_counts')}")
                if summary.get('rare_label_counts'):
                    st.caption(f"Preview rare-label coverage: {summary.get('rare_label_counts')}")
            diagnostics = st.session_state.get('external_fetch_diagnostics', [])
            if diagnostics:
                with st.expander("External fetch diagnostics", expanded=True):
                    for diag in diagnostics:
                        st.caption(diag)
            st.session_state.pop('last_external_fetch_message', None)

        preview_actions_col1, preview_actions_col2 = st.columns(2)
        with preview_actions_col1:
            if os.path.exists("combined_external_append_ready.csv"):
                st.download_button(
                    label="🧩 Download Preview Append-Ready External Batch",
                    data=Path("combined_external_append_ready.csv").read_bytes(),
                    file_name="combined_external_append_ready.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True
                )
        with preview_actions_col2:
            append_disabled = not os.path.exists("combined_external_append_ready.csv")
            if st.button("➕ Append Preview Batch to Master Dataset", icon=":material/add_circle:", use_container_width=True, disabled=append_disabled):
                preview_df = pd.read_csv("combined_external_append_ready.csv")
                append_rows_to_master_dataset(
                    preview_df,
                    latest_batch_filename="latest_external_append_batch.csv",
                    success_prefix="External preview append"
                )
                st.rerun()

        if os.path.exists("external_teacher_candidates.csv"):
            teacher_target = st.slider("Teacher-label external candidates", min_value=50, max_value=1000, value=300, step=50, help="Gemini teacher-label pass for the external unlabeled candidate pool.")
            if st.button("🧠 Teacher-Label External Candidates & Append", icon=":material/psychology:", use_container_width=True):
                with st.spinner("Teacher-labeling external candidates with Gemini 3.1 Pro..."):
                    candidate_df = pd.read_csv("external_teacher_candidates.csv")
                    existing_df = pd.read_csv("master_training_data.csv") if os.path.exists("master_training_data.csv") else pd.DataFrame()
                    if not existing_df.empty and 'text_hash' not in existing_df.columns and 'text' in existing_df.columns:
                        existing_df['text_hash'] = existing_df['text'].astype(str).apply(build_text_hash)
                    existing_hashes = set(existing_df['text_hash'].dropna().astype(str).tolist()) if ('text_hash' in existing_df.columns and not existing_df.empty) else set()

                    labeled_external_df = teacher_label_external_candidates_df(
                        candidate_df,
                        sample_size=teacher_target,
                        excluded_hashes=existing_hashes
                    )
                    if labeled_external_df is not None and not labeled_external_df.empty:
                        labeled_external_df = labeled_external_df.copy()
                        labeled_external_df['sample_weight'] = EXTERNAL_DATASET_DEFAULT_WEIGHTS['counsel_chat_teacher']
                        labeled_external_df['source_dataset'] = 'counsel_chat_teacher'
                        labeled_external_df['teacher_data_version'] = 'external_teacher_labeled_v2'
                        labeled_external_df.to_csv("latest_external_teacher_labeled_batch.csv", index=False)
                        append_rows_to_master_dataset(
                            labeled_external_df,
                            latest_batch_filename="latest_external_teacher_labeled_batch.csv",
                            success_prefix="External teacher-labeled append"
                        )
                        st.rerun()

        if st.session_state.get('last_external_fetch_message'):
            st.info(st.session_state['last_external_fetch_message'])
            st.session_state.pop('last_external_fetch_message', None)

        # --- RESTORE MASTER DATASET (CLOUD FIX) ---
        uploaded_file = st.file_uploader("Restore Master Dataset (CSV)", type="csv", help="If the cloud server reboots, upload your last downloaded master dataset here to restore it.")
        if uploaded_file is not None:
            try:
                restored_df = pd.read_csv(uploaded_file)
                if 'text_hash' not in restored_df.columns and 'text' in restored_df.columns:
                    restored_df['text_hash'] = restored_df['text'].astype(str).apply(build_text_hash)
                restored_df.to_csv("master_training_data.csv", index=False)
                st.success(f"✅ Restored {len(restored_df)} rows to the server!")
            except Exception as e:
                st.error(f"Failed to restore file: {e}")

        thresholds_upload = st.file_uploader(
            "Restore Recommended Thresholds (JSON)",
            type="json",
            help="Optional manual override: upload recommended_thresholds.json from Colab if you want to replace the active model's auto-loaded cutoffs."
        )
        if thresholds_upload is not None:
            try:
                uploaded_thresholds = validate_threshold_map(json.load(thresholds_upload))
                st.session_state['active_label_thresholds'] = uploaded_thresholds
                st.session_state['threshold_source_mode'] = 'manual_upload'
                st.session_state['threshold_source_label'] = 'Manual override from uploaded recommended_thresholds.json'
                st.session_state['threshold_source_model_id'] = st.session_state.get('active_model_id', 'SamLowe/roberta-base-go_emotions')
                with open(THRESHOLD_FILE_PATH, 'w') as f:
                    json.dump(uploaded_thresholds, f, indent=2)
                st.success("✅ Applied manual threshold override.")
            except Exception as e:
                st.error(f"Failed to restore threshold file: {e}")

        if st.session_state.get('last_training_append_message'):
            st.success(st.session_state['last_training_append_message'])
            bucket_mix = st.session_state.get('last_training_bucket_mix', {})
            if bucket_mix:
                st.caption(f"Sample mix: {bucket_mix}")
            stats = st.session_state.get('last_generation_stats', {})
            if stats:
                st.caption(
                    f"Generation diagnostics -> sampled: {stats.get('sampled_candidates', 0):,}, "
                    f"teacher-valid: {stats.get('teacher_valid_rows', 0):,}, "
                    f"teacher-invalid/failed: {stats.get('teacher_invalid_or_failed', 0):,}, "
                    f"new-unique-candidates: {stats.get('new_unique_candidate_rows', 0):,}, "
                    f"written-to-master: {stats.get('rows_written_to_master', 0):,}"
                )
            st.session_state.pop('last_training_append_message', None)
            st.session_state.pop('last_training_bucket_mix', None)

        if os.path.exists("master_training_data.csv"):
            df_master = pd.read_csv("master_training_data.csv")
            sampler_cols = [col for col in ['sampler_bucket', 'teacher_data_version', 'text_hash'] if col in df_master.columns]
            dataset_msg = f"📦 **Master Dataset:** {len(df_master)} rows ready."
            if sampler_cols:
                dataset_msg += f" Metadata available: {', '.join(sampler_cols)}."
            st.info(dataset_msg)
            
            dataset_bytes = Path("master_training_data.csv").read_bytes()
            st.download_button(
                label="2️⃣ Download Master Dataset",
                data=dataset_bytes,
                file_name="full_trainingdata.csv",
                mime="text/csv",
                icon=":material/download:",
                use_container_width=True
            )

            if os.path.exists("latest_generated_training_batch.csv"):
                latest_batch_bytes = Path("latest_generated_training_batch.csv").read_bytes()
                st.download_button(
                    label="🧪 Download Latest Generated Batch",
                    data=latest_batch_bytes,
                    file_name="latest_generated_training_batch.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True
                )


            if os.path.exists("exact_append_ready.csv"):
                st.download_button(
                    label="🌐 Download GoEmotions Exact Batch",
                    data=Path("exact_append_ready.csv").read_bytes(),
                    file_name="exact_append_ready.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True
                )

            if os.path.exists("mapped_append_ready.csv"):
                st.download_button(
                    label="🧩 Download Mapped Supplemental Batch",
                    data=Path("mapped_append_ready.csv").read_bytes(),
                    file_name="mapped_append_ready.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True
                )

            if os.path.exists("external_teacher_candidates.csv"):
                st.download_button(
                    label="🧠 Download External Teacher Candidates",
                    data=Path("external_teacher_candidates.csv").read_bytes(),
                    file_name="external_teacher_candidates.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True
                )

            if os.path.exists("latest_external_teacher_labeled_batch.csv"):
                st.download_button(
                    label="✨ Download Latest External Teacher-Labeled Batch",
                    data=Path("latest_external_teacher_labeled_batch.csv").read_bytes(),
                    file_name="latest_external_teacher_labeled_batch.csv",
                    mime="text/csv",
                    icon=":material/download:",
                    use_container_width=True
                )
                
            st.markdown("""
            <a href="https://colab.research.google.com/drive/19e9AOuYaFZ1zhXpY_yH0SkzR_o1RYkKd" target="_blank" style="text-decoration: none;">
                <button style="width: 100%; background-color: #fbbc05; color: black; border: none; padding: 10px; border-radius: 5px; font-weight: bold; cursor: pointer; margin-bottom: 10px;">
                    3️⃣ Launch Google Colab Fine-Tuning ↗️
                </button>
            </a>
            """, unsafe_allow_html=True)

        st.markdown("### ⚙️ 4. Deploy Custom Model")
        hf_username = st.text_input("HuggingFace Username", value="", help="Enter your HF username to auto-fetch your fine-tuned models.")
        
        available_models = fetch_custom_models(hf_username)
        active_model_id = st.selectbox("Active Model", available_models, help="Select a model to instantly hot-swap the backend AI.")
        st.session_state['active_model_id'] = active_model_id
        sync_thresholds_for_active_model(active_model_id)
        render_threshold_source_badge()
        st.caption("Reset Threshold Source discards any local/manual threshold override and returns the app to the selected model\'s default threshold behavior. Refresh Thresholds from Active Model re-fetches recommended_thresholds.json from the selected Hugging Face model repo if that file changed after the app loaded.")

        col_reset, col_refresh = st.columns(2)
        with col_reset:
            if st.button("Reset Threshold Source", use_container_width=True, help="Use this after a manual JSON upload if you want to go back to the active model or base fallback thresholds."):
                try:
                    if os.path.exists(THRESHOLD_FILE_PATH):
                        os.remove(THRESHOLD_FILE_PATH)
                except Exception:
                    pass
                fetch_model_repo_thresholds.clear()
                sync_thresholds_for_active_model(active_model_id, force_refresh=True)
                st.rerun()
        with col_refresh:
            if st.button("Refresh Thresholds from Active Model", use_container_width=True, help="Use this when the active model repo has a newer recommended_thresholds.json and you want to pull it again without restarting the app."):
                fetch_model_repo_thresholds.clear()
                sync_thresholds_for_active_model(active_model_id, force_refresh=True)
                st.rerun()
        
        if active_model_id != "SamLowe/roberta-base-go_emotions":
            st.success(f"🟢 Custom Model Active")

    st.sidebar.markdown("---")
    st.sidebar.markdown("## :material/security: Scam / Fraud Detection")
    st.sidebar.markdown("""
        <div class="metric-definition" style="background-color: #fff7e6; border-left-color: #ff9f1a;">
            <h5>Two-Stage Pipeline</h5>
            <ul>
                <li><b>Tier 1:</b> Deterministic regex / heuristic screening for common scam, phishing, financial-fraud, and off-platform solicitation patterns (live weekly escalation: <b>severity-based score ≥ 5</b>, with shortlinks treated as high severity).</li>
                <li><b>Scam scan window:</b> For any selected week, fraud/scam detection runs on that week <b>plus the prior 3 weeks</b> under the hood.</li>
                <li><b>Tier 2:</b> Gemini trust-and-safety review using Chain-of-Thought reasoning to separate predatory behavior from legitimate networking/support.</li>
                <li><b>Tier 2.5:</b> Military-context override for medium-confidence scam calls (PCS/deployment/EFMP/VA context) to reduce false positives for military/veteran families.</li>
                <li><b>A/B Threshold Policy:</b> Sidebar A/B controls now affect both escalation decisions and severity tiering.</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    last_run = st.session_state.get("scam_last_run", "Not run yet")
    scam_review_mode = st.session_state.get("scam_review_mode", "Tier 1 + Tier 2 (Gemini)")
    scam_review_error = st.session_state.get("scam_review_error", "")
    current_flags = 0
    if st.session_state.get("results"):
        current_flags = len(st.session_state.results.get("scam_flags", []) or[])

    st.sidebar.markdown(f"""
        <div class="metric-definition" style="background-color: #f3f3f3; border-left-color: #666;">
            <h5>Run status</h5>
            <p><b>Last run:</b> {last_run}</p>
            <p><b>Review mode:</b> {scam_review_mode}</p>
            <p><b>Flagged for review:</b> {current_flags}</p>
        </div>
    """, unsafe_allow_html=True)

    if scam_review_error:
        st.sidebar.caption(f"⚠️ Gemini status: {scam_review_error}")

    scam_window_hint = st.session_state.get("scam_scan_window_hint")
    if scam_window_hint:
        tooltip_text = html.escape(
            "Fraud/scam scanning runs on the selected week plus the prior 3 weeks under the hood. "
            "Severe-emotion detection remains scoped to the selected week as before."
        )
        st.sidebar.markdown(
            f"<span title='{tooltip_text}'>🗂️ Scam window active</span>",
            unsafe_allow_html=True,
        )

    with st.sidebar.expander("QA overrides", expanded=False, icon=":material/bug_report:"):
        st.checkbox(
            "Include staff/admin posts in weekly scam scan",
            key="scam_include_trusted_users",
            help="QA-only override. Keep OFF in production to avoid flagging official BSF/admin posts.",
        )
        if st.session_state.get("scam_include_trusted_users"):
            st.caption("🧪 QA override ON: trusted-user bypass disabled for scam scanning.")

    with st.sidebar.expander("Test scam detector", expanded=False, icon=":material/search:"):
        test_text = st.text_area("Paste a comment to test", key="scam_test_text", height=120)
        st.caption("Uses the live weekly Tier-1 rule: severity-based escalation at score ≥ 5, with shortlinks treated as high severity.")
        run_llm = st.checkbox("Use Gemini verification (Tier 2)", value=True, key="scam_test_use_llm")
        if st.button("Run test", key="scam_test_btn", icon=":material/play_arrow:"):
            if not test_text.strip(): st.warning("Paste a comment to test.")
            else:
                h = scam_heuristic_scan(test_text)
                tier1_triggered, severity_level = classify_fraud_severity(h)
                if tier1_triggered:
                    st.error(f"🚨 **Tier 1 (Heuristics) Triggered** — Severity: {severity_level}")
                else:
                    st.success("✅ **Tier 1 Passed**")
                st.json(h)
                if run_llm and (h.get("heuristic_score", 0) >= 1 or h.get("matched_terms")):
                    gem_raw = _gemini_scam_classify_cached(test_text)
                    try:
                        gem = json.loads(gem_raw)
                        if gem.get("is_scam"): st.error("🚨 **Tier 2 (Gemini) Flagged as Scam**")
                        else: st.success("✅ **Tier 2 (Gemini) Passed**")
                        st.json(gem)
                    except:
                        st.warning("Could not parse Gemini response.")
                        st.write(gem_raw)

    with st.sidebar.expander("Run curated benchmark", expanded=False, icon=":material/lab_profile:"):
        benchmark_corpus = load_scam_benchmark_corpus()
        st.caption(f"{len(benchmark_corpus)} paraphrased real-world examples spanning smishing, loan, job, gift-card, prize, investment, and military-context-safe controls.")
        benchmark_limit = st.slider(
            "Items to evaluate",
            min_value=8,
            max_value=max(len(benchmark_corpus), 8),
            value=min(24, max(len(benchmark_corpus), 8)),
            step=4,
            key="scam_benchmark_limit"
        ) if benchmark_corpus else 0
        benchmark_use_pipeline = st.checkbox(
            "Use full production pipeline (Tier 1 + Gemini + Tier 2.5)",
            value=False,
            key="scam_benchmark_use_pipeline",
            help="Tier 1 only is fast. Full pipeline measures the same review path used during weekly analysis."
        )
        if st.button("Run benchmark", key="scam_benchmark_btn", icon=":material/analytics:", use_container_width=True):
            if not benchmark_corpus:
                st.warning("Benchmark corpus file not found.")
            else:
                with st.spinner("Running curated scam benchmark..."):
                    st.session_state["scam_benchmark_results"] = evaluate_scam_benchmark(
                        max_items=benchmark_limit,
                        run_full_pipeline=benchmark_use_pipeline,
                    )

        benchmark_results = st.session_state.get("scam_benchmark_results")
        if benchmark_results:
            active_metrics = benchmark_results["pipeline_metrics"] if benchmark_results.get("run_full_pipeline") and benchmark_results.get("pipeline_metrics") else benchmark_results["tier1_metrics"]
            active_label = "Full pipeline" if benchmark_results.get("run_full_pipeline") else "Tier 1 only"
            st.caption(f"Last benchmark mode: {active_label}")
            bm1, bm2 = st.columns(2)
            with bm1:
                st.metric("Precision", f"{active_metrics['precision_pct']}%")
                st.metric("Recall", f"{active_metrics['recall_pct']}%")
            with bm2:
                st.metric("Specificity", f"{active_metrics['specificity_pct']}%")
                st.metric("False +", f"{active_metrics['false_positive_rate_pct']}%")
            if benchmark_results.get("pipeline_metrics"):
                st.caption(
                    f"Tier 1 recall {benchmark_results['tier1_metrics']['recall_pct']}% → "
                    f"Full pipeline recall {benchmark_results['pipeline_metrics']['recall_pct']}% | "
                    f"Audit queue items: {benchmark_results.get('audit_queue_count', 0)}"
                )
            st.dataframe(benchmark_results["category_summary"], use_container_width=True, height=220, hide_index=True)
            st.download_button(
                "Download benchmark CSV",
                data=benchmark_results["results_csv"],
                file_name="scam_benchmark_results.csv",
                mime="text/csv",
                icon=":material/download:",
                use_container_width=True,
            )

    if analyze_button:
        st.session_state.analysis_run = True
        st.session_state.results = {}
        st.session_state["audit_queue"] = []  # Reset stale queue from prior run
        
        if st.session_state.sanity_log:
             current_time = datetime.now().strftime('%H:%M:%S')
             st.session_state.sanity_log.append({"timestamp": "---", "comment_index": "---", "comment_preview": f"🌊 NEW RUN [{current_time}] 🌊", "flagged_for": "", "raw_response": "", "interpreted_label": "SEPARATOR", "status": "Separator"})
        
        year = int(selected_year)
        week_num = extract_week_number(selected_week_str)
        active_model_id = st.session_state.get('active_model_id', "SamLowe/roberta-base-go_emotions")

        progress_ui = AnalysisProgressUI()
        progress_ui.set_overall_message(":material/search: Loading weekly interactions from Snowflake...")

        df = load_data_for_week(week_num, year, show_progress=False)
        if df.empty:
            st.error("No data found for the selected week.")
            st.session_state.analysis_run = False
            st.stop()

        progress_ui.set_overall_message(f":material/search: Loaded {len(df)} interactions. Running all 4 analysis steps...")

        # Step 1: Emotion Analysis (UI version, NOT cached)
        df_em = analyze_emotions_with_ui(df, model_id=active_model_id, progress_ui=progress_ui)
        avg = df_em.mean() if not df_em.empty else pd.Series()
        score = compute_health_score(avg)

        # Step 2: Sanity Check (Moved up to run immediately after Emotion Analysis)
        concerns, initially_flagged = detect_severe_concerns(df_em, df, silent=False, progress_ui=progress_ui)
        st.session_state['last_initial_flagged_count'] = len(initially_flagged)

        # Step 3 & 4: Scam Check
        scam_detection_df, scam_window_weeks = load_scam_detection_window(week_num, year, window_weeks=4)
        if scam_detection_df is None or scam_detection_df.empty:
            # Safety fallback to selected-week-only behavior if historical pulls are unavailable.
            scam_detection_df = df.copy()
            scam_window_weeks = [f"W{week_num}/{year}"]

        include_trusted_users = bool(st.session_state.get("scam_include_trusted_users", False))
        scam_flags = detect_scam_concerns(
            scam_detection_df,
            silent=False,
            progress_ui=progress_ui,
            include_trusted_users=include_trusted_users,
        )
        st.session_state['scam_scan_window_hint'] = " · ".join(scam_window_weeks) if scam_window_weeks else f"W{week_num}/{year}"
        st.session_state['last_scam_tier1_count'] = count_scam_tier1_candidates(
            scam_detection_df,
            include_trusted_users=include_trusted_users,
        )
        
        # Optional safety-audit path (hidden by default in dashboard UX).
        false_negatives = []
        fn_stats = {
            "sample_size": 0,
            "passed_total": 0,
            "false_negatives_detected": 0,
            "false_negative_rate": 0.0,
        }
        if SHOW_FALSE_NEGATIVE_AUDIT_SECTIONS:
            flagged_indices = set(flag.get("index") for flag in scam_flags)
            false_negatives, fn_stats = detect_false_negatives_sampling(scam_detection_df, flagged_indices, sample_size=50, silent=False)

            # Persist false-negative rate to rolling session history for trend chart.
            # Upsert by (week, year) so re-running the same week updates rather than duplicates.
            if "fn_rate_history" not in st.session_state:
                st.session_state["fn_rate_history"] = []
            _fn_entry = {
                "label": f"W{week_num}/{year}",
                "week": week_num,
                "year": year,
                "false_negative_rate": fn_stats.get("false_negative_rate", 0.0),
                "false_negatives_detected": fn_stats.get("false_negatives_detected", 0),
                "sample_size": fn_stats.get("sample_size", 0),
                "scam_flags": len(scam_flags),
            }
            _existing_idx = next(
                (i for i, h in enumerate(st.session_state["fn_rate_history"])
                 if h["week"] == week_num and h["year"] == year),
                None
            )
            if _existing_idx is not None:
                st.session_state["fn_rate_history"][_existing_idx] = _fn_entry
            else:
                st.session_state["fn_rate_history"].append(_fn_entry)

        # Slack alert for critical-severity scam flags
        if scam_flags:
            critical_flags = categorize_scam_severity(scam_flags).get("critical", [])
            if critical_flags:
                slack_sent = send_slack_alert(critical_flags, week_num, year)
                if slack_sent:
                    st.toast(f"🔔 Slack alert sent for {len(critical_flags)} CRITICAL flag(s).", icon="🔔")

        progress_ui.finalize("✅ Pipeline Execution Complete!")

        # Historical data (Cached version, NO UI)
        prev_week_date = datetime.strptime(f'{year}-W{week_num:02d}-1', "%G-W%V-%u").date() - timedelta(weeks=1)
        prev_year, prev_week, _ = prev_week_date.isocalendar()
        df_prev = load_data_for_week(prev_week, prev_year, show_progress=False)
        df_em_prev = analyze_emotions_cached(df_prev, model_id=active_model_id)
        avg_prev = df_em_prev.mean() if not df_em_prev.empty else pd.Series()
        score_prev = compute_health_score(avg_prev)
        
        mild_moderate_comments = get_mild_moderate_comments(df_em, df, top_n=50)
        st.session_state['scam_last_run'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        st.session_state.results = {
            "df": df, "df_em": df_em, "avg": avg, "score": score, 
            "concerns": concerns, "initially_flagged": initially_flagged,
            "score_prev": score_prev, "mild_moderate_comments": mild_moderate_comments,
            "scam_flags": scam_flags, "false_negatives": false_negatives, "fn_stats": fn_stats,
            "scam_scan_window_weeks": scam_window_weeks,
            "scam_scan_comment_count": len(scam_detection_df),
            "week_num": week_num, "year": year
        }
        st.rerun() 

    if st.session_state.analysis_run and st.session_state.results:
        res = st.session_state.results
        required_keys =["score", "concerns", "initially_flagged", "week_num", "year", "avg", "df"]
        if not all(key in res for key in required_keys):
            st.error("An error occurred during analysis. Please try again.")
            st.session_state.analysis_run = False
            st.session_state.results = {}
            st.stop()
        
        score, score_prev, concerns, initially_flagged, mild_moderate_comments, week_num, year, avg, df, df_em = (
            res['score'], res.get('score_prev'), res['concerns'], res['initially_flagged'], 
            res['mild_moderate_comments'], res['week_num'], res['year'], res['avg'], res['df'], res['df_em']
        )
        scam_flags = res.get('scam_flags',[])
        active_model_id = st.session_state.get('active_model_id', "SamLowe/roberta-base-go_emotions")

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            gemini_verified_only = [c for c in concerns if c.get('verdict_source') == 'gemini_true']
            selected_week_data = {
                "score": score if score is not None else 0,
                "severe_count": len(gemini_verified_only),
                "comment_count": len(df),
                "scam_count": len(scam_flags)
            }
            _save_weekly_verified_counts(
                year, week_num,
                selected_week_data['severe_count'],
                selected_week_data['scam_count'],
                selected_week_data['comment_count'],
                selected_week_data['score'],
                active_model_id,
            )
            _hires_plot_config = {
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": f"bsf_trend_week{week_num}_{year}",
                    "scale": 3,
                },
                "displaylogo": False,
            }
            st.plotly_chart(
                create_dual_trend_plot(week_num, year, selected_week_data, active_model_id),
                use_container_width=True,
                config=_hires_plot_config,
            )
        with col2:
            _radar_fig = create_radar_plot(avg)
            _radar_buf = io.BytesIO()
            _radar_fig.savefig(_radar_buf, format="png", dpi=300, bbox_inches="tight")
            _radar_buf.seek(0)
            st.markdown(
                """
                <style>
                div[data-testid="column"]:has(.bsf-radar-anchor) { position: relative; }
                div[data-testid="column"]:has(.bsf-radar-anchor) [data-testid="stDownloadButton"] {
                    position: absolute; top: 4px; right: 44px; z-index: 5;
                    opacity: 0.12; transition: opacity 0.2s ease; width: auto;
                }
                div[data-testid="column"]:has(.bsf-radar-anchor):hover [data-testid="stDownloadButton"] { opacity: 0.9; }
                div[data-testid="column"]:has(.bsf-radar-anchor) [data-testid="stDownloadButton"] button {
                    padding: 2px 6px; min-height: 0; height: 26px;
                    background: transparent; border: 1px solid #bbb; color: #555;
                    font-size: 12px; line-height: 1;
                }
                </style>
                <div class="bsf-radar-anchor"></div>
                """,
                unsafe_allow_html=True,
            )
            st.download_button(
                ":material/download:",
                data=_radar_buf,
                file_name=f"bsf_radar_week{week_num}_{year}.png",
                mime="image/png",
                key=f"radar_dl_{week_num}_{year}",
                help="Download radar chart (high-res PNG)",
            )
            st.pyplot(_radar_fig, dpi=200, use_container_width=True)
            plt.close(_radar_fig)
        
        _emotion_trend_config = {
            "toImageButtonOptions": {
                "format": "png",
                "filename": f"bsf_emotion_trend_week{week_num}_{year}",
                "scale": 3,
            },
            "displaylogo": False,
        }
        st.plotly_chart(
            create_emotion_trend_chart(week_num, year, active_model_id),
            use_container_width=True,
            config=_emotion_trend_config,
        )
        
        st.subheader(":material/assignment: Weekly Summary")
        summary = make_summary(score, avg, concerns, len(df), df_em, df=df, prev_score=score_prev, week=week_num, year=year, scam_count=len(scam_flags))
        html_summary = summary.replace("\n\n", "<br><br>").replace("\n", "<br>")
        st.markdown(f'<div style="border: 2px solid #d0d0d0; border-radius: 10px; padding: 25px; background-color: #f8f9fa;">{html_summary}</div>', unsafe_allow_html=True)
        
        if st.button("Email Severe Report", icon=":material/mail:"):
            if concerns: send_email_alert(concerns, week_num, year)
            else: st.warning("No severe items were found to email.")
        
        st.markdown("---")
        st.subheader(":material/security: Scam / Fraud Flags (Risk-Stratified)")
        scam_window_weeks = res.get('scam_scan_window_weeks', [f"W{week_num}/{year}"])
        scam_window_count = int(res.get('scam_scan_comment_count', len(df)))
        if scam_window_weeks:
            window_tooltip = html.escape(
                f"Window used for scam detection: {' · '.join(scam_window_weeks)}. "
                f"Total interactions scanned: {scam_window_count}."
            )
            st.markdown(
                f"<span title='{window_tooltip}'>🛈 Scam/fraud scan uses rolling 4-week window</span>",
                unsafe_allow_html=True,
            )
        
        # Initialise HITL feedback log in session
        if "scam_feedback_log" not in st.session_state:
            st.session_state["scam_feedback_log"] = []

        def _render_flag_card(flag, severity_label: str, key_prefix: str):
            """Render one flag card with HITL feedback controls."""
            title_user = normalize_identity_value(flag.get('commenter_name', 'N/A'), fallback='Unknown')
            idx        = flag.get('index', 'N/A')
            conf       = flag.get('confidence', None)
            conf_txt   = f"{conf:.2f}" if isinstance(conf, (int, float)) else "N/A"
            heur_score = flag.get('heuristic_score', 'N/A')
            matched    = ', '.join((flag.get('matched_terms') or [])[:6]) or 'N/A'
            mil_ctx    = flag.get('military_context_pattern', '')

            expanded = severity_label == "CRITICAL"
            label = f"{'⚠️ ' if expanded else ''}{title_user} (Item #{idx}) — Conf: {conf_txt}"
            with st.expander(f"**{label}**", expanded=expanded):
                c1, c2 = st.columns(2)
                with c1:
                    display_email = normalize_identity_value(flag.get('commenter_email', 'N/A'), fallback='N/A')
                    detected_email = normalize_identity_value(flag.get('detected_contact_email', 'N/A'), fallback='N/A')
                    st.markdown(f"**Scan Week:** {normalize_identity_value(flag.get('scan_week', 'Selected week'), fallback='Selected week')}")
                    st.markdown(f"**Author Email:** {display_email}")
                    if detected_email != 'N/A':
                        st.markdown(f"**Email found in text:** {detected_email}")
                    st.markdown(f"**Community:** {normalize_identity_value(flag.get('community_name', 'N/A'), fallback='N/A')}")
                    st.markdown(f"**Heuristic score:** {heur_score} | **Terms:** _{matched}_")
                    if mil_ctx:
                        st.caption(f"🎖️ Military context: {mil_ctx}")
                with c2:
                    st.markdown(f"**Indicators:** {', '.join((flag.get('indicators') or [])[:5]) or 'N/A'}")
                    reasons = flag.get('reasons', []) or []
                    if reasons:
                        st.markdown("**AI Reasoning:**")
                        for r in reasons[:3]:
                            st.markdown(f"- {html.escape(str(r))}")
                st.markdown("**Text:**")
                st.write(flag.get('comment', ''))
                
                # HITL Feedback Buttons
                st.markdown("---")
                st.caption("📝 Staff Review — mark your decision to improve the model:")
                fb_key = f"{key_prefix}_{idx}"
                existing_fb = next((f["verdict"] for f in st.session_state["scam_feedback_log"] if f["flag_index"] == idx), None)
                if existing_fb:
                    st.success(f"✅ Already marked: **{existing_fb}**")
                else:
                    btn_c1, btn_c2, btn_c3 = st.columns(3)
                    with btn_c1:
                        if st.button("✅ Confirm Scam", key=f"hitl_confirm_{fb_key}", use_container_width=True):
                            st.session_state["scam_feedback_log"].append({
                                "flag_index": idx, "verdict": "CONFIRMED_SCAM",
                                "commenter_name": title_user, "confidence": conf,
                                "ts": datetime.now().isoformat(),
                                "comment_preview": str(flag.get("comment", ""))[:120],
                            })
                            st.rerun()
                    with btn_c2:
                        if st.button("🚫 False Positive", key=f"hitl_fp_{fb_key}", use_container_width=True):
                            st.session_state["scam_feedback_log"].append({
                                "flag_index": idx, "verdict": "FALSE_POSITIVE",
                                "commenter_name": title_user, "confidence": conf,
                                "ts": datetime.now().isoformat(),
                                "comment_preview": str(flag.get("comment", ""))[:120],
                            })
                            st.rerun()
                    with btn_c3:
                        if st.button("🔍 Needs Review", key=f"hitl_rev_{fb_key}", use_container_width=True):
                            st.session_state["scam_feedback_log"].append({
                                "flag_index": idx, "verdict": "NEEDS_HUMAN_REVIEW",
                                "commenter_name": title_user, "confidence": conf,
                                "ts": datetime.now().isoformat(),
                                "comment_preview": str(flag.get("comment", ""))[:120],
                            })
                            st.rerun()

        if scam_flags:
            severity_buckets = categorize_scam_severity(scam_flags)
            prioritized_flags = sorted(
                [flag for bucket in severity_buckets.values() for flag in bucket],
                key=lambda flag: (
                    int(flag.get("review_priority", 99) or 99),
                    -float(flag.get("confidence", 0.0) or 0.0),
                    -int(flag.get("heuristic_score", 0) or 0),
                    int(flag.get("index", 10**9) or 10**9) if isinstance(flag.get("index"), (int, np.integer)) else 10**9,
                )
            )
            
            # Summary metrics by severity
            col_crit, col_high, col_med, col_low = st.columns(4)
            with col_crit:
                st.metric("🔴 CRITICAL", len(severity_buckets["critical"]), delta="Urgent review", delta_color="inverse")
            with col_high:
                st.metric("🟠 HIGH", len(severity_buckets["high"]), delta="Priority review", delta_color="inverse")
            with col_med:
                st.metric("🟡 MEDIUM", len(severity_buckets["medium"]), delta="Review queue")
            with col_low:
                st.metric("🔵 LOW", len(severity_buckets["low"]), delta="Audit tracking")
            
            # Slack alert status
            critical_flags_display = severity_buckets["critical"]
            if critical_flags_display:
                st.caption(f"🔔 Slack alert sent for {len(critical_flags_display)} CRITICAL flag(s) — check your #bsf-alerts channel.")

            st.caption("Sorted top-to-bottom by severity priority, Gemini confidence, then heuristic score.")
            with st.container(height=560):
                for flag in prioritized_flags:
                    _render_flag_card(flag, str(flag.get("severity", "LOW")), "flag")

            # HITL Feedback log summary
            fb_log = st.session_state.get("scam_feedback_log", [])
            if fb_log:
                st.markdown("---")
                st.markdown("#### 📋 Staff Feedback Log (This Session)")
                confirmed   = [f for f in fb_log if f["verdict"] == "CONFIRMED_SCAM"]
                false_pos   = [f for f in fb_log if f["verdict"] == "FALSE_POSITIVE"]
                needs_rev   = [f for f in fb_log if f["verdict"] == "NEEDS_HUMAN_REVIEW"]
                fb_c1, fb_c2, fb_c3 = st.columns(3)
                with fb_c1: st.metric("✅ Confirmed Scams", len(confirmed))
                with fb_c2: st.metric("🚫 False Positives", len(false_pos))
                with fb_c3: st.metric("🔍 Needs Review", len(needs_rev))
                
                fb_df = pd.DataFrame(fb_log)[["ts", "flag_index", "commenter_name", "verdict", "confidence", "comment_preview"]]
                fb_df.columns = ["Timestamp", "Item #", "User", "Verdict", "Confidence", "Comment Preview"]
                st.dataframe(fb_df, use_container_width=True, height=180)
                
                fb_csv = fb_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Export Feedback CSV (for model retraining)",
                    data=fb_csv, file_name=f"scam_hitl_feedback_W{week_num}_{year}.csv",
                    mime="text/csv", icon=":material/download:", use_container_width=False
                )
        else:
            st.info("No scam/fraud candidates were flagged for review this week.")
        
        # Display audit queue (items suppressed by Tier 2 filtering for transparency)
        audit_queue = st.session_state.get("audit_queue", [])
        if audit_queue:
            st.markdown("---")
            st.subheader(":material/info: Audit Queue (Tier 1 Matched, Tier 2 Suppressed)")
            st.caption(f"Items detected by heuristics but Gemini unclear (low confidence): {len(audit_queue)}")
            st.caption("These are **not** immediate action items, but mark any you think are real scams to improve the model.")
            with st.container(height=340):
                for aq_idx, item in enumerate(audit_queue[:10]):
                    user   = normalize_identity_value(item.get('commenter_name', 'N/A'), fallback='Unknown')
                    idx    = item.get('index', 'N/A')
                    reason = item.get('audit_reason', 'Audit tracking')
                    heur   = item.get('heuristic_score', 'N/A')
                    conf   = item.get('confidence', 'N/A')
                    terms  = ', '.join((item.get('matched_terms') or [])[:6]) or 'N/A'
                    mil    = item.get('military_context_pattern', '')
                    with st.expander(f"**{user}** (Item #{idx}) — {reason}"):
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption(f"Heuristic Score: {heur} | Gemini Confidence: {conf}")
                            st.caption(f"Matched terms: _{terms}_")
                            detected_email = normalize_identity_value(item.get('detected_contact_email', 'N/A'), fallback='N/A')
                            if detected_email != 'N/A':
                                st.caption(f"Email found in text: {detected_email}")
                        with c2:
                            if mil:
                                st.caption(f"🎖️ Military context: {mil}")
                            st.markdown(f"**Community:** {normalize_identity_value(item.get('community_name', 'N/A'), fallback='N/A')}")
                        st.markdown("**Text:**")
                        st.write(item.get('comment', ''))
                        
                        # HITL: staff can mark suppressed items as actual scams
                        aq_fb_key = f"aq_{idx}_{aq_idx}"
                        existing = next((f["verdict"] for f in st.session_state["scam_feedback_log"] if f["flag_index"] == f"aq_{idx}"), None)
                        if existing:
                            st.success(f"✅ Marked: {existing}")
                        else:
                            ab1, ab2 = st.columns(2)
                            with ab1:
                                if st.button("🚨 Mark as Real Scam", key=f"aqmark_{aq_fb_key}", use_container_width=True):
                                    st.session_state["scam_feedback_log"].append({
                                        "flag_index": f"aq_{idx}", "verdict": "CONFIRMED_SCAM (was suppressed)",
                                        "commenter_name": user, "confidence": conf,
                                        "ts": datetime.now().isoformat(),
                                        "comment_preview": str(item.get("comment", ""))[:120],
                                    })
                                    st.rerun()
                            with ab2:
                                if st.button("✅ Correctly Suppressed", key=f"aqok_{aq_fb_key}", use_container_width=True):
                                    st.session_state["scam_feedback_log"].append({
                                        "flag_index": f"aq_{idx}", "verdict": "TRUE_NEGATIVE (correctly suppressed)",
                                        "commenter_name": user, "confidence": conf,
                                        "ts": datetime.now().isoformat(),
                                        "comment_preview": str(item.get("comment", ""))[:120],
                                    })
                                    st.rerun()
            if len(audit_queue) > 10:
                st.caption(f"... and {len(audit_queue) - 10} more items in audit queue")
        
        # Display false negative safety audit results
        false_negatives = res.get("false_negatives", [])
        fn_stats = res.get("fn_stats", {})
        
        if SHOW_FALSE_NEGATIVE_AUDIT_SECTIONS and (fn_stats.get("false_negative_rate", 0) > 0 or false_negatives):
            st.markdown("---")
            st.subheader("⚠️ Safety Audit: False Negatives (Scams That Passed)")
            st.caption("Random sampling of 50 posts that passed detection, re-checked for missed scams")
            
            col_fn1, col_fn2, col_fn3, col_fn4 = st.columns(4)
            with col_fn1:
                st.metric("Sample Size", fn_stats.get("sample_size", 0), help="Posts checked from non-flagged pool")
            with col_fn2:
                st.metric("Missed Scams", fn_stats.get("false_negatives_detected", 0), delta="High confidence", delta_color="inverse")
            with col_fn3:
                st.metric("False Neg. Rate", f"{fn_stats.get('false_negative_rate', 0)}%", help="Percentage of sampled posts that were scams")
            with col_fn4:
                st.metric("Passed Total", fn_stats.get("passed_total", 0), help="Posts that were not flagged")
            
            if false_negatives:
                st.markdown("**⚠️ Missed Scams (High Confidence):**")
                with st.container(height=250):
                    for fn in false_negatives[:3]:
                        user = fn.get('commenter_name', 'N/A')
                        idx = fn.get('index', 'N/A')
                        conf = fn.get('confidence', 0)
                        with st.expander(f"🔴 {user} (Item #{idx}) — Confidence: {conf:.2f}"):
                            st.caption(f"**Community:** {fn.get('community_name', 'N/A')}")
                            st.markdown(f"**Indicators:** {', '.join(fn.get('indicators', [])[:5])}")
                            st.markdown(f"**Reasoning:** {fn.get('reasoning', 'High-confidence scam')}")
                            st.write(fn.get('comment', ''))
                if len(false_negatives) > 3:
                    st.caption(f"... and {len(false_negatives) - 3} more missed scams in this sample")

        # Weekly false-negative rate trend chart
        fn_history = st.session_state.get("fn_rate_history", [])
        if SHOW_FALSE_NEGATIVE_AUDIT_SECTIONS and fn_history:
            st.markdown("---")
            st.subheader(":material/trending_up: Safety Trend: False-Negative Rate Over Time")
            st.plotly_chart(create_fn_rate_trend_chart(fn_history), use_container_width=True)
            st.caption("Accumulated across all weekly analyses in this session. Export HITL feedback CSV to persist across sessions.")

        st.markdown("---")
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader(":material/warning: Severe Flagged Items")
            st.caption("ℹ️ Tier-2 (Gemini-verified True) severe items only. Items rejected by the AI sanity check — or included only via recall override — are hidden.")
            severe_display_items = sorted(
                [c for c in concerns if c.get('verdict_source') == 'gemini_true'],
                key=lambda x: x.get('total_negative', 0),
                reverse=True,
            )
            with st.container(height=600):
                if severe_display_items:
                    for concern in severe_display_items:
                        with st.expander(f"**User:** {concern.get('commenter_name', 'N/A')} (Item #{concern['index']})"):
                            st.markdown(f"**Email:** {concern.get('commenter_email', 'N/A')}")
                            st.markdown(f"**Community:** {concern.get('community_name', 'N/A')}")
                            st.markdown(concern['comment'])
                            with st.popover("🔍 Details", use_container_width=False):
                                st.caption("Status: ✅ Gemini-verified")
                                st.caption(
                                    f"Active severe components ({concern.get('negative_component_count', 0)}): "
                                    f"{', '.join(concern.get('negative_components', [])) or 'None'}"
                                )
                else:
                    st.info("No Gemini-verified severe items for this week.")

        with col4:
            st.subheader(":material/error_outline: Mild–Moderate Items")
            st.caption("ℹ️ Same unit logic as severe items: cumulative load is a sum across negative labels, while average intensity is bounded and easier to interpret.")
            with st.container(height=600):
                if mild_moderate_comments:
                    for comment in mild_moderate_comments:
                        with st.expander(f"**User:** {comment.get('commenter_name', 'N/A')} (Item #{comment['index']})"):
                            st.markdown(f"**Email:** {comment.get('commenter_email', 'N/A')}")
                            st.markdown(f"**Community:** {comment.get('community_name', 'N/A')}")
                            st.markdown(comment['comment'])
                            with st.popover("🔍 Details", use_container_width=False):
                                mild_m1, mild_m2, mild_m3 = st.columns(3)
                                mild_m1.metric(
                                    ":material/warning: Dominant Emotion",
                                    f"{comment['emotion']} ({comment['score']}%)",
                                    help=(
                                        "Single strongest negative/contextual emotion for this item. "
                                        "Unit: percent intensity on a bounded 0–100 scale."
                                    )
                                )
                                mild_m2.metric(
                                    "Cumulative Negative Load",
                                    f"{comment['total_negative']} pts",
                                    help=(
                                        "Exact formula: 100 × Σ score(e) across mild-path negative emotions. "
                                        "Unit is cumulative points (additive multi-label load), so values may exceed 100."
                                    )
                                )
                                mild_m3.metric(
                                    "Avg Negative Intensity",
                                    f"{comment.get('avg_negative_intensity', 0.0)}%",
                                    help=(
                                        "Exact formula: (100 × Σ active negative scores) / (# active negative emotions). "
                                        "Bounded 0–100 for easier interpretation."
                                    )
                                )
                                st.caption(
                                    f"Active negative components ({comment.get('negative_component_count', 0)}): "
                                    f"{', '.join(comment.get('negative_components', [])) or 'None'}"
                                )
                else:
                    st.info("No mild–moderate worry items found.")

    display_sanity_check_log()

    st.sidebar.markdown("---")
    if st.sidebar.button("🔒 Log Out", key="logout_btn", use_container_width=True):
        clear_authenticated_session()
        st.rerun()

def main():
    st.set_page_config(page_title="BSF Community Wellness Dashboard", page_icon="logo.png", layout="wide", initial_sidebar_state="expanded")

    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if st.session_state.get('authenticated'):
        run_dashboard()
    else:
        render_login_page()

if __name__ == "__main__":
    main()
