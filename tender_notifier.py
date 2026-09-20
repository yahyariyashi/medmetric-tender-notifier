import os
import time
import json
import requests
import threading
import multiprocessing
import traceback
import hmac
import psycopg2
from datetime import datetime, timezone, timedelta
from flask import Flask, request
from playwright.sync_api import sync_playwright
import urllib3
from openai import OpenAI

def _call_groq_with_retry(client, system_instruction, prompt, temperature=None, max_retries=None, base_delay=None):
    """
    Wraps Groq chat.completions.create with retries + exponential backoff.

    Env configuration (all optional):
      GROQ_MODEL          default openai/gpt-oss-120b
                          alternatives: openai/gpt-oss-20b (faster/lighter)
      GROQ_TEMPERATURE    default 0.2  (0 = more deterministic JSON)
      GROQ_MAX_TOKENS     default 2048 (completion size cap)
      GROQ_MAX_RETRIES    default 3
      GROQ_RETRY_DELAY    default 2     (seconds; multiplied by attempt)
    """
    model_name = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")
    if temperature is None:
        try:
            temperature = float(os.environ.get("GROQ_TEMPERATURE", "0.2"))
        except ValueError:
            temperature = 0.2
    if max_retries is None:
        try:
            max_retries = int(os.environ.get("GROQ_MAX_RETRIES", "3"))
        except ValueError:
            max_retries = 3
    if base_delay is None:
        try:
            base_delay = float(os.environ.get("GROQ_RETRY_DELAY", "2"))
        except ValueError:
            base_delay = 2
    try:
        max_tokens = int(os.environ.get("GROQ_MAX_TOKENS", "2048"))
    except ValueError:
        max_tokens = 2048

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * attempt  # 2s, 4s, 6s...
                print(f"⚠️ Groq call failed (attempt {attempt}/{max_retries}): {e} — retrying in {delay}s", flush=True)
                time.sleep(delay)
    raise last_error


def _call_gemini_with_retry(system_instruction, prompt, temperature=0.2, max_retries=3, base_delay=2):
    """
    Calls Google Gemini generateContent REST API with retries.
    Uses GEMINI_API_KEY and GEMINI_MODEL (default gemini-flash-latest).
    No extra pip package required — plain requests.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    model = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model}:generateContent?key={api_key}"
    )
    body = {
        "system_instruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "responseMimeType": "application/json",
        },
    }
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=body, timeout=90)
            if r.status_code != 200:
                raise RuntimeError(f"Gemini HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            # Normalize to a Groq-like object with .choices[0].message.content
            text_out = ""
            for cand in data.get("candidates") or []:
                parts = ((cand.get("content") or {}).get("parts")) or []
                for p in parts:
                    text_out += p.get("text") or ""
            if not text_out.strip():
                raise RuntimeError(f"Gemini empty response: {str(data)[:300]}")

            class _Msg:
                def __init__(self, content):
                    self.content = content

            class _Choice:
                def __init__(self, content):
                    self.message = _Msg(content)

            class _Resp:
                def __init__(self, content):
                    self.choices = [_Choice(content)]

            return _Resp(text_out)
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                delay = base_delay * attempt
                print(f"⚠️ Gemini call failed (attempt {attempt}/{max_retries}): {e} — retrying in {delay}s", flush=True)
                time.sleep(delay)
    raise last_error




# Keywords used to locate the parts of a detail page worth keeping when the
# full captured text (up to 6000 chars) is too long to send to the AI in
# full. A blind [:N] truncation was cutting off the closing date and
# eligibility/constraints info entirely whenever a page's nav/breadcrumb/menu
# text ran long before the real content — which is exactly why those fields
# kept coming back empty even though the scraper WAS capturing the text.
_CLOSING_DATE_HINTS = [
    "closing date", "submission deadline", "bid submission", "deadline",
    "closing time", "bid closing", "last date", "due date", "valid until",
    "validity", "opening date",
]
_CONSTRAINT_HINTS = [
    "eligib", "bid bond", "bid security", "minimum requirement", "experience",
    "similar contract", "qualification", "capital", "registration",
    "local agent", "representative", "certificate", "license", "requirement",
]


def _build_detail_snippet(detail_text: str, max_chars: int = 1200) -> str:
    """
    Instead of blindly slicing the first N characters (which was silently
    dropping the closing date and constraints whenever they appeared later
    in the page than the cutoff), keep the front of the page for
    title/scope context PLUS a window of text around any closing-date or
    constraint-related keyword found anywhere in the full captured text.
    This makes it far more likely the AI actually sees the relevant
    sentence instead of nav/menu boilerplate.
    """
    if not detail_text:
        return ""
    if len(detail_text) <= max_chars:
        return detail_text

    lower = detail_text.lower()
    front_budget = max_chars // 2
    front = detail_text[:front_budget]

    windows = []
    remaining_budget = max_chars - front_budget
    for hint in _CLOSING_DATE_HINTS + _CONSTRAINT_HINTS:
        if remaining_budget <= 0:
            break
        idx = lower.find(hint)
        if idx == -1 or idx < front_budget:
            continue  # already covered by the front slice, or not present
        start = max(0, idx - 60)
        end = min(len(detail_text), idx + 240)
        window = detail_text[start:end]
        if window not in windows:
            windows.append(window)
            remaining_budget -= len(window)

    return front + "\n[...]\n" + "\n[...]\n".join(windows) if windows else front


def _build_batch_prompt(chunk_with_indices):
    """Builds the system + user prompt for one sub-batch of (index, candidate) pairs."""
    entries = []
    for i, c in chunk_with_indices:
        snippet = _build_detail_snippet(c.get('detail_text', ''))
        entries.append(
            f"[{i}] Source: {c['source']} | Title: {c['title']} | Ref No: {c.get('ref_no', '')}\n"
            f"Detail: {snippet}"
        )
    joined_entries = "\n\n".join(entries)

    system_instruction = (
        "You are an expert procurement analyst for medMETRIC Healthcare Service PLC, "
        "an Ethiopian biomedical engineering and medical equipment technical-services provider. Their scope "
        "covers ALL of the following as EQUALLY relevant — do not treat maintenance/service as the deciding "
        "factor or require it to be present:\n"
        "1) Medical equipment corrective and preventive maintenance & service contracts (including hemodialysis "
        "systems like B.Braun Dialog+ and SWS-4000A), RO/water treatment systems, CSSD & sterilization equipment "
        "(e.g. Rivamed or Aquaboss), and maintenance of medical imaging equipment.\n"
        "2) Straight supply/procurement of medical equipment with NO maintenance component attached — including "
        "International Competitive Bidding (ICB) for medical equipment and general medical equipment supply. "
        "A tender that is ONLY 'procurement/supply of medical equipment' with no service attached IS relevant — "
        "do NOT reject it just because it lacks a maintenance or service component.\n"
        "3) CPD training, coaching, workshops, or consultancy services related to medical equipment or biomedical "
        "engineering — e.g. medical device maintenance training for hospital technicians, biomedical workshop "
        "organization, technical consultancy on medical equipment. These count as relevant even though they are "
        "not equipment supply or a maintenance contract.\n"
        "They do NOT supply medicines, must check since they don't work on "
        "pharmaceuticals, vaccines, or laboratory-only equipment/reagents/consumables — mark "
        "those NOT relevant even if they mention \"medical\" in passing. Generic non-technical services (e.g. "
        "environmental/social impact assessments, civil works, furniture, IT) remain NOT relevant even if the "
        "underlying project involves a hospital.\n\n"
        "You will be given a numbered list of tenders, each with a Detail excerpt taken from the tender's "
        "own detail page (it may include a front section plus separate windows of text pulled from around "
        "date/eligibility keywords elsewhere on the page, joined by [...] markers — treat each window as "
        "real page text, not as connected prose).\n\n"
        "Analyze EACH tender independently and return a JSON object with a single key \"tenders\" whose "
        "value is an array — one object per tender, covering every index given, with EXACTLY this shape:\n"
        '{"tenders": [{'
        '"index": 0, '
        '"relevant": true, '
        '"match_score": 85, '
        '"reason": "max 12 words why it matches or not", '
        '"procurement_object": "the object/type of procurement as stated on the page, e.g. \'Supply and installation of hemodialysis machines\'", '
        '"constraints": "key eligibility/bid requirements actually stated in the text — e.g. bid bond amount, minimum years of similar experience, required certifications, local representation requirement. '
        'If the provided excerpt does not contain this info, respond exactly with \'Not stated in provided text\' — do NOT say \'None\' or \'None identified\', since that implies you confirmed there are none.", '
        '"closing_date": "the bid submission deadline / closing date (and time, if given) EXACTLY as it appears in the text, including the date format used. '
        'If no such date appears anywhere in the excerpt, respond exactly with \'Not stated in provided text\'. Do not confuse this with a publish date, site-visit date, or clarification-request date."'
        "}, ...]}\n"
        "Respond with ONLY that JSON object — no markdown fences, no commentary."
    )
    prompt = f"Tenders to analyze:\n\n{joined_entries}"
    return system_instruction, prompt



def batch_analyze_tenders(candidates: list):
    """
    Analyze candidates with SEPARATE Groq free-tier keys:
      - 2merkato  → GROQ_API_KEY
      - egp       → GROQ_API_KEY_EGP  (falls back to GROQ_API_KEY if unset)

    Each source is chunked independently so one source cannot exhaust the
    other key's quota. Returns {global_index: analysis_dict} or None if
    every call failed for every item.
    """
    if not candidates:
        return {}

    CHUNK_SIZE = int(os.environ.get("AI_CHUNK_SIZE", "3"))

    def _run_chunks(indexed_subset, label: str, api_key: str):
        """indexed_subset: list of (global_index, candidate)."""
        if not indexed_subset:
            return {}, False
        if not api_key:
            print(f"⚠️ No Groq key for {label} — skipping AI analysis for this source", flush=True)
            return {}, False

        client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
        chunks = [indexed_subset[i:i + CHUNK_SIZE] for i in range(0, len(indexed_subset), CHUNK_SIZE)]
        results = {}
        any_ok = False

        for chunk_num, chunk in enumerate(chunks, start=1):
            local_pairs = [(local_i, c) for local_i, (_, c) in enumerate(chunk)]
            global_by_local = {local_i: global_i for local_i, (global_i, _) in enumerate(chunk)}
            system_instruction, prompt = _build_batch_prompt(local_pairs)
            try:
                response = _call_groq_with_retry(client, system_instruction, prompt)
                raw = (response.choices[0].message.content or "").strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                    raw = raw.strip()
                parsed = json.loads(raw)
                items = parsed.get("tenders", []) if isinstance(parsed, dict) else parsed
                for item in items:
                    local_idx = item.get("index")
                    if local_idx is None:
                        continue
                    global_idx = global_by_local.get(local_idx)
                    if global_idx is not None:
                        item = dict(item)
                        item["index"] = global_idx
                        results[global_idx] = item
                any_ok = True
                print(f"✅ {label} sub-batch {chunk_num}/{len(chunks)} ok ({len(chunk)} tenders)", flush=True)
            except Exception as e:
                print(f"⚠️ {label} sub-batch {chunk_num}/{len(chunks)} failed: {e}", flush=True)

            if chunk_num < len(chunks):
                time.sleep(5)

        return results, any_ok

    merkato = [(i, c) for i, c in enumerate(candidates) if c.get("source") == "2merkato"]
    egp = [(i, c) for i, c in enumerate(candidates) if c.get("source") == "egp"]
    other = [(i, c) for i, c in enumerate(candidates) if c.get("source") not in ("2merkato", "egp")]

    key_merkato = os.environ.get("GROQ_API_KEY", "")
    key_egp = os.environ.get("GROQ_API_KEY_EGP") or key_merkato

    print(
        f"🤖 AI split: {len(merkato)} × Groq/2merkato, {len(egp)} × Groq/eGP"
        + (f", {len(other)} × Groq/other" if other else "")
        + ("" if os.environ.get("GROQ_API_KEY_EGP") else " [eGP using same key as 2merkato — set GROQ_API_KEY_EGP for a second quota]"),
        flush=True,
    )

    all_results = {}
    any_ok = False

    r, ok = _run_chunks(merkato, "groq-merkato", key_merkato)
    all_results.update(r)
    any_ok = any_ok or ok

    r, ok = _run_chunks(egp, "groq-egp", key_egp)
    all_results.update(r)
    any_ok = any_ok or ok

    r, ok = _run_chunks(other, "groq-other", key_merkato)
    all_results.update(r)
    any_ok = any_ok or ok

    return all_results if any_ok else None



urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)

# ================== CONFIGURATION ==================
EVOLUTION_BASE = os.environ.get("EVOLUTION_BASE", "https://medmetric-evolution.onrender.com")
INSTANCE_NAME = os.environ.get("INSTANCE_NAME", "Tender-Notifier.")

# SECURITY: no hardcoded fallback — a real API key must never live in source
# code, especially in a public repo. If it's missing, we log a clear warning
# at startup rather than silently sending requests with an empty key.
GLOBAL_API_KEY = os.environ.get("GLOBAL_API_KEY")
if not GLOBAL_API_KEY:
    print("⚠️ GLOBAL_API_KEY is not set — WhatsApp sends will fail until it's configured in Render's environment variables.", flush=True)

# SECURITY/PRIVACY: no hardcoded phone number — recipients must be supplied
# via the WHATSAPP_NUMBERS env var (comma-separated).
WHATSAPP_NUMBERS = [
    n.strip() for n in os.environ.get("WHATSAPP_NUMBERS", "").split(",") if n.strip()
]
if not WHATSAPP_NUMBERS:
    print("⚠️ WHATSAPP_NUMBERS is not set — no recipients configured.", flush=True)

# Shared secret required to trigger a manual scan via /test-check. Without
# this, anyone who discovers the public Render URL could trigger scans on
# demand, burning your Groq quota and spamming your WhatsApp numbers.
TEST_CHECK_TOKEN = os.environ.get("TEST_CHECK_TOKEN")

MERKATO_USER = os.environ.get("MERKATO_USER")
MERKATO_PASS = os.environ.get("MERKATO_PASS")

# CORRECT domain — the working tenders app, not www.2merkato.com
MERKATO_BASE = "https://tender.2merkato.com"
MERKATO_LOGIN_URL = f"{MERKATO_BASE}/login"
MERKATO_TENDERS_URL = f"{MERKATO_BASE}/tenders"
MERKATO_MAX_PAGES = int(os.environ.get("MERKATO_MAX_PAGES", "4"))

# The eGP domain has moved/varied before (production.egp.gov.et vs plain
# egp.gov.et, and there's a real chance it could live at something like
# egp.ppa.gov.et in the future). Making this a single env-var-driven
# constant means switching domains is a Render dashboard change, not a
# code change — every URL below (login, bids listing, home-navigation
# recovery, detail-page link resolution) derives from this one value.
# .rstrip("/") guards against a trailing slash in the env var producing
# doubled slashes in every derived URL below (e.g. "https://egp.gov.et/"
# + "/egp/login" would otherwise become ".../egp.gov.et//egp/login").
EGP_BASE = os.environ.get("EGP_BASE", "https://production.egp.gov.et").rstrip("/")
EGP_LOGIN_URL = f"{EGP_BASE}/egp/login"
EGP_BIDS_URL = f"{EGP_BASE}/egp/bids/all"
EGP_USER = os.environ.get("EGP_USER")
EGP_PASS = os.environ.get("EGP_PASS")
EGP_ORG_NAME = os.environ.get("EGP_ORG_NAME", "medMETRIC Healthcare Service PLC")

# OpenID Connect / IdentityServer settings discovered from the frontend bundle
# and /.well-known/openid-configuration. The SPA itself uses the password grant
# with these exact client credentials, so we can do the same and completely
# bypass the browser-based login (and the inspect-not-allowed protection).
EGP_TOKEN_URL = f"{EGP_BASE}/auth/connect/token"
EGP_CLIENT_ID = os.environ.get("EGP_CLIENT_ID", "skoruba_identity_admin")
EGP_CLIENT_SECRET = os.environ.get("EGP_CLIENT_SECRET", "skoruba_admin_client_secret")
EGP_SCOPES = "openid profile email roles erp_api offline_access"

# Search terms fed one at a time into eGP's own built-in table search box —
# far more reliable than scraping every row across 13+ pages.
# Rebuilt around medMETRIC's actual service lines (medmetrichealthcare.com):
# maintenance/service contracts, RO water treatment, CSSD/sterilization,
# and calibration — not just dialysis. "laboratory" intentionally dropped —
# lab-only tenders are explicitly out of scope.
EGP_SEARCH_TERMS = [
    "Procurement of Maintenance and Repair Service", "Procurement of Medical Equipment and Supplies", "hemodialysis", "dialysis",
    "medical equipment", "sterilization", "water treatment", "Reverse osmosis", "filters",
    "x-ray", "ultrasound", "medical imaging", "membrane", "ICB", "International Competitive Bid",
    "B Braun", "Dialog+", "ዲያሊሲስ",
]

# Search terms fed one at a time into 2merkato's own "Search by keyword" filter
# panel — mirrors the EGP_SEARCH_TERMS approach above. 2merkato's general
# listing has 25,000+ pages, so paging through the first few of those (the
# old approach) only ever saw whatever happened to be posted most recently
# across ALL categories, not specifically medical/biomedical tenders. Reusing
# the site's own keyword search narrows results down to a handful of pages
# per term, the same way it does on eGP.
MERKATO_SEARCH_TERMS = [
    "hemodialysis", "dialysis", "medical equipment", "sterilization",
    "water treatment", "reverse osmosis", "x-ray", "ultrasound",
    "medical imaging", "biomedical", "CSSD", "autoclave", "ICB",
    "medical equipment maintenance", "hospital equipment",
    "B Braun", "Dialog+", "ዲያሊሲስ",
]

# How many result pages to walk PER search term (not per whole catalog) —
# a filtered keyword search returns far fewer pages than the full listing,
# so a small number here is plenty and keeps the scan fast.
MERKATO_MAX_PAGES_PER_TERM = int(os.environ.get("MERKATO_MAX_PAGES_PER_TERM", "3"))

# Any ONE of these alone is specific enough to trigger a match.
# Includes medMETRIC's actual named service lines (CSSD, RO/water treatment,
# calibration, spare parts, biomedical engineering) so tenders aren't
# under-scored just because they're not dialysis-specific.
STRONG_KEYWORDS = [
    "ICB", "hemodialysis", "dialysis", "bbraun", "philips", "SWS",
    "x-ray", "xray", "ultrasound", "CT Scan", "autoclave", "Water treatment",
    "Procurement of Maintenance and Repair Service", "cssd", "Procurement of Medical Equipment and Supplies",
    "reverse osmosis", "ro system", "SPHMMC", "technical service",
    "biomedical engineering", "medical imaging", "calibration",
    "EPSA", "medical equipment", "hospital equipment",
    "medical device", "የህክምና ጥገና",
]

# Generic medical-adjacent words — only count if paired with an equipment/
# procurement-type word in the same title (avoids matching HR/insurance/
# consulting tenders that merely mention "health"). "laboratory" removed —
# lab-only tenders are handled by HARD_EXCLUDE_TERMS below instead.
MEDICAL_CONTEXT = ["medical", "health", "hospital", "biomedical", "clinical"]
EQUIPMENT_CONTEXT = ["equipment", "supplies", "supply", "device", "machine",
                     "ICB", "corrective maintenance", "maintenance", "repair", "procurement of medical equipment",
                     "calibration", "medical equipment installation", "servicing",
                     "Maintenance and Repair Service", "medical consultancy" ]

# Always excluded regardless of context — these categories are never
# relevant to Medmetric no matter what else appears in the title.
# Laboratory-only tenders and pure medicine/pharmaceutical supply tenders
# (e.g. "RDF Medicines...") are explicitly out of scope — medMETRIC is a
# biomedical engineering/technical-service company, not a drug supplier.
HARD_EXCLUDE_TERMS = [
    "vehicle", "toyota", "car ", "motorbike", "insurance", "life insurance",
    "term life", "gpa", "spare part", "construction","road"
    "laboratory", "lab reagent", "reagent", "lab equipment",
    "medicine", "medicines", "pharmaceutical", "pharmaceuticals",
    "drug", "drugs", "vaccine", "vaccines", "rdf medicine", "rdf medicines"
]

# Excluded UNLESS the title also shows clear medical + equipment context —
# generic "consultancy services" for HR/finance/etc. should be skipped, but
# "medical equipment consultancy services" should NOT be, since Medmetric
# offers exactly that
CONTEXTUAL_EXCLUDE_TERMS = ["car", "consulting firm"]


def is_relevant_tender(title):
    title_lower = title.lower()

    if any(term in title_lower for term in HARD_EXCLUDE_TERMS):
        return False

    has_medical_context = any(term in title_lower for term in MEDICAL_CONTEXT)
    has_equipment_context = any(term in title_lower for term in EQUIPMENT_CONTEXT)

    if any(term in title_lower for term in CONTEXTUAL_EXCLUDE_TERMS):
        if not (has_medical_context and has_equipment_context):
            return False
        # else: it's medical-equipment consultancy — fall through, don't exclude

    if any(term in title_lower for term in STRONG_KEYWORDS):
        return True

    return has_medical_context and has_equipment_context


# ==================== PERSISTENT DEDUP (Postgres) ====================
DATABASE_URL = os.environ.get("DATABASE_URL")

# Reuse a single connection across the whole scan instead of opening/closing
# a new one per tender — repeated SSL handshakes were adding real memory/CPU
# overhead on Render's free tier and were the likely cause of the scan
# stalling or getting silently killed partway through page 1.
_db_conn = None


def get_db_connection():
    global _db_conn
    if _db_conn is not None:
        try:
            # Cheap liveness check — reuse the connection if it's still good
            cur = _db_conn.cursor()
            cur.execute("SELECT 1")
            cur.close()
            return _db_conn
        except Exception:
            try:
                _db_conn.close()
            except Exception:
                pass
            _db_conn = None

    _db_conn = psycopg2.connect(DATABASE_URL, sslmode="require", connect_timeout=10)
    return _db_conn


def init_db():
    """Run once at startup to ensure the dedup table exists."""
    if not DATABASE_URL:
        print("⚠️ DATABASE_URL not set — falling back to in-memory dedup (will reset on restart)", flush=True)
        return
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS notified_tenders (
                tender_id TEXT PRIMARY KEY,
                notified_at TIMESTAMP DEFAULT NOW()
            )
        """)
        conn.commit()
        cur.close()
        print("✅ Connected to Postgres and verified notified_tenders table", flush=True)
    except Exception as e:
        print(f"❌ Postgres init failed, falling back to in-memory dedup: {e}", flush=True)


# In-memory fallback used only if DATABASE_URL is missing or unreachable —
# keeps the app functional (just without persistence) rather than crashing
_MEMORY_FALLBACK = set()


def is_already_notified(tender_id: str) -> bool:
    if not DATABASE_URL:
        return tender_id in _MEMORY_FALLBACK
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1 FROM notified_tenders WHERE tender_id = %s", (tender_id,))
        result = cur.fetchone()
        cur.close()
        return result is not None
    except Exception as e:
        print(f"⚠️ DB check failed for {tender_id}, using memory fallback: {e}", flush=True)
        return tender_id in _MEMORY_FALLBACK


def mark_as_notified(tender_id: str):
    if not DATABASE_URL:
        _MEMORY_FALLBACK.add(tender_id)
        return
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO notified_tenders (tender_id) VALUES (%s) ON CONFLICT DO NOTHING",
            (tender_id,)
        )
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"⚠️ DB insert failed for {tender_id}, using memory fallback: {e}", flush=True)
        _MEMORY_FALLBACK.add(tender_id)


def send_whatsapp(message):
    if not GLOBAL_API_KEY or not WHATSAPP_NUMBERS:
        print("❌ Cannot send WhatsApp message: GLOBAL_API_KEY or WHATSAPP_NUMBERS is not configured.", flush=True)
        return

    url = f"{EVOLUTION_BASE}/message/sendText/{INSTANCE_NAME}"
    headers = {"Content-Type": "application/json", "apikey": GLOBAL_API_KEY}
    for number in WHATSAPP_NUMBERS:
        payload = {"number": number, "text": message}
        try:
            r = requests.post(url, json=payload, headers=headers, timeout=10)
            # PRIVACY: mask most of the phone number in logs rather than
            # printing it in full — logs can end up shared/exported more
            # widely than intended
            masked = number[:5] + "***" + number[-2:] if len(number) > 7 else "***"
            print(f"✅ WhatsApp Status ({masked}): {r.status_code}", flush=True)
        except Exception as e:
            print(f"❌ Send error: {e}", flush=True)



# ==================== ENGINE: 2MERKATO (Playwright) ====================
def scrape_2merkato(candidates: list):
    print(f"[{datetime.now()}] 🔍 Running 2merkato Engine (Playwright)...", flush=True)
    found = 0

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"]
            )
            context = browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            )
            page = context.new_page()

            # --- Login (optional — only if credentials provided) ---
            if MERKATO_USER and MERKATO_PASS:
                try:
                    page.goto(MERKATO_LOGIN_URL, timeout=30000, wait_until="domcontentloaded")
                    page.wait_for_selector("input[type='password']", timeout=20000)

                    field_info = page.eval_on_selector_all(
                        "input",
                        "els => els.map(e => ({type: e.type, name: e.name, id: e.id, placeholder: e.placeholder}))"
                    )
                    print(f"🔎 Login page input fields detected: {field_info}", flush=True)

                    email_field = page.locator(
                        "input[type='email'], input[name*='email' i], input[name*='user' i], "
                        "input[id*='email' i], input[id*='user' i], input[placeholder*='email' i]"
                    ).first
                    pass_field = page.locator("input[type='password']").first

                    if email_field.count() > 0 and pass_field.count() > 0:
                        email_field.fill(MERKATO_USER)
                        pass_field.fill(MERKATO_PASS)

                        submit_btn = page.locator(
                            "button[type='submit'], button:has-text('Login'), button:has-text('Sign in'), "
                            "button:has-text('Log in'), input[type='submit']"
                        ).first
                        submit_btn.click()
                        page.wait_for_timeout(4000)
                        print(f"✅ Submitted login, current URL: {page.url}", flush=True)
                    else:
                        print(f"⚠️ Could not confidently identify email/username field among: {field_info}", flush=True)
                except Exception as e:
                    print(f"⚠️ Login attempt failed, continuing without auth: {e}", flush=True)

            page.goto(MERKATO_TENDERS_URL, timeout=30000, wait_until="domcontentloaded")
            page.wait_for_timeout(2000)

            search_input = page.locator(
                "input[placeholder*='Search by keyword' i], input[placeholder*='keyword' i]"
            ).first

            if search_input.count() == 0:
                header_clickables = page.eval_on_selector_all(
                    "header *, nav *",
                    "els => els.slice(0, 40).map(e => ({tag: e.tagName, "
                    "aria: e.getAttribute('aria-label'), cls: e.className, "
                    "text: (e.innerText || '').trim().slice(0, 30)}))"
                )
                print(f"🔎 2merkato search input not immediately visible. Header elements: {header_clickables}", flush=True)

                toggle = page.locator(
                    "[aria-label*='search' i], button:has-text('Search'), "
                    "header svg, nav svg, [class*='search-icon' i], [class*='search-toggle' i]"
                ).first
                if toggle.count() > 0:
                    try:
                        toggle.click(timeout=5000)
                        page.wait_for_timeout(1000)
                        print("✅ Clicked a probable search-toggle icon in the header", flush=True)
                    except Exception as e:
                        print(f"⚠️ Could not click search-toggle candidate: {e}", flush=True)

                search_input = page.locator(
                    "input[placeholder*='Search by keyword' i], input[placeholder*='keyword' i]"
                ).first

            if search_input.count() == 0:
                print("❌ 2merkato search-by-keyword input never appeared — falling back to the old first-page listing so this scan isn't a total loss.", flush=True)
                links = page.locator("a[href*='/tenders/']").all()
                print(f"Fallback listing page: found {len(links)} raw tender links", flush=True)
                found += _process_merkato_links(links, context, candidates, seen_this_scan=set())
                browser.close()
                print(f"2merkato engine complete. Collected {found} candidate matches for AI review.", flush=True)
                return found

            print("✅ Found 2merkato 'Search by keyword' input — proceeding with keyword-driven search", flush=True)

            seen_this_scan = set()
            for term in MERKATO_SEARCH_TERMS:
                try:
                    if search_input.count() == 0 or not search_input.is_visible():
                        toggle = page.locator(
                            "[aria-label*='search' i], button:has-text('Search'), "
                            "header svg, nav svg, [class*='search-icon' i], [class*='search-toggle' i]"
                        ).first
                        if toggle.count() > 0:
                            try:
                                toggle.click(timeout=5000)
                                page.wait_for_timeout(1000)
                            except Exception:
                                pass
                        search_input = page.locator(
                            "input[placeholder*='Search by keyword' i], input[placeholder*='keyword' i]"
                        ).first

                    search_input.click()
                    search_input.fill("")
                    page.wait_for_timeout(200)
                    search_input.fill(term)

                    url_before = page.url
                    search_input.press("Enter")
                    page.wait_for_timeout(2000)

                    if page.url == url_before:
                        filter_btn = page.locator("button:has-text('Filter')").first
                        if filter_btn.count() > 0:
                            try:
                                filter_btn.click(timeout=5000, force=True)
                            except Exception as e:
                                print(f"⚠️ Forced Filter click also failed for '{term}': {e}", flush=True)

                    page.wait_for_timeout(2500)
                    print(f"➡️ 2merkato search '{term}' submitted, current URL: {page.url}", flush=True)

                    for page_num in range(1, MERKATO_MAX_PAGES_PER_TERM + 1):
                        if page_num > 1:
                            next_btn = page.locator(
                                "button:has-text('Next'), a:has-text('Next'), [aria-label*='next' i]"
                            ).first
                            if next_btn.count() == 0:
                                break
                            try:
                                next_btn.click(timeout=5000)
                                page.wait_for_timeout(2000)
                            except Exception:
                                break

                        links = page.locator("a[href*='/tenders/']").all()
                        print(f"2merkato search '{term}' page {page_num}: found {len(links)} raw tender links", flush=True)
                        if not links:
                            break

                        found += _process_merkato_links(links, context, candidates, seen_this_scan)

                except Exception as e:
                    print(f"⚠️ 2merkato search for '{term}' failed: {e}", flush=True)
                    continue

            browser.close()

        print(f"2merkato engine complete. Collected {found} candidate matches for AI review.", flush=True)
        return found

    except Exception as e:
        print(f"❌ 2merkato extraction engine down: {e}", flush=True)
        return 0


def _get_merkato_card_status(link) -> str:
    try:
        container = link.locator("xpath=ancestor::*[contains(., 'Bidding')][1]").first
        if container.count() == 0:
            return "unknown"
        card_text = container.inner_text(timeout=3000)
    except Exception:
        return "unknown"

    lower = card_text.lower()
    idx = lower.find("bidding")
    if idx == -1:
        return "unknown"
    window = lower[idx: idx + 40]
    if "closed" in window:
        return "closed"
    if "open" in window:
        return "open"
    return "unknown"


def _process_merkato_links(links, context, candidates: list, seen_this_scan: set) -> int:
    found = 0
    for link in links:
        try:
            title_text = link.inner_text().strip()
            href = link.get_attribute("href")
            if not title_text or not href:
                continue

            full_link = href if href.startswith("http") else f"{MERKATO_BASE}{href}"

            if full_link in seen_this_scan:
                continue
            seen_this_scan.add(full_link)

            if is_relevant_tender(title_text):
                tender_id = f"merkato_{full_link.rstrip('/').split('/')[-1]}"
                if not is_already_notified(tender_id):
                    status = _get_merkato_card_status(link)
                    if status == "closed":
                        print(f"⏭️ Skipping closed 2merkato tender (Bidding: Closed): {title_text}", flush=True)
                        mark_as_notified(tender_id)
                        continue

                    # NOTE: do NOT mark_as_notified() here — same reasoning as
                    # the eGP engine. This candidate hasn't been AI-reviewed or
                    # sent yet; marking it now would permanently bury it if the
                    # AI batch fails or genuinely (but wrongly) rejects it.
                    found += 1

                    detail_text = ""
                    detail_page = None
                    try:
                        detail_page = context.new_page()
                        detail_page.goto(full_link, timeout=20000, wait_until="domcontentloaded")
                        try:
                            detail_page.wait_for_load_state("networkidle", timeout=10000)
                        except Exception:
                            pass
                        for poll_attempt in range(5):
                            detail_page.wait_for_timeout(1000)
                            try:
                                candidate_text = detail_page.locator("body").inner_text(timeout=5000)
                            except Exception:
                                candidate_text = ""
                            if len(candidate_text) > 300:
                                detail_text = candidate_text
                                break
                            detail_text = candidate_text
                        detail_text = detail_text[:6000]
                        print(f"📄 Captured {len(detail_text)} chars from 2merkato detail page", flush=True)
                    except Exception as e:
                        print(f"⚠️ Could not read 2merkato detail page text: {e}", flush=True)
                    finally:
                        if detail_page is not None:
                            try:
                                detail_page.close()
                            except Exception:
                                pass

                    candidates.append({
                        "id": tender_id,
                        "source": "2merkato",
                        "title": title_text,
                        "ref_no": "",
                        "detail_text": detail_text,
                        "link": full_link,
                    })
        except Exception:
            continue
    return found



def _egp_api_get(path_and_query: str, timeout: int = 30):
    """GET from the public eGP gateway (/po-gw/...). Returns parsed JSON or None."""
    url = f"{EGP_BASE}/po-gw{path_and_query}" if path_and_query.startswith("/") else f"{EGP_BASE}/po-gw/{path_and_query}"
    try:
        r = requests.get(url, timeout=timeout, verify=False)
        if r.status_code != 200:
            print(f"⚠️ eGP API {r.status_code} for {path_and_query[:80]}", flush=True)
            return None
        return r.json()
    except Exception as e:
        print(f"⚠️ eGP API error for {path_and_query[:60]}: {e}", flush=True)
        return None


def _egp_build_detail_text(item: dict) -> str:
    """Build a text snippet the AI can use from the rich API item (no detail-page fetch needed)."""
    parts = []
    parts.append(f"Title: {item.get('lotName', '')}")
    parts.append(f"Description: {item.get('lotDescription', '')}")
    parts.append(f"Reference: {item.get('procurementReferenceNo') or item.get('lotReferenceNo', '')}")
    parts.append(f"Category: {item.get('procurementCategory', '')}")
    parts.append(f"Method: {item.get('method', '')}")
    parts.append(f"Market: {item.get('marketPlace', '')}")
    parts.append(f"Procuring Entity: {item.get('procuringEntity', '')}")
    parts.append(f"Submission Deadline / Closing Date: {item.get('submissionDeadline', '')}")
    parts.append(f"Invitation Date: {item.get('invitationDate', '')}")
    parts.append(f"Is Submittable: {item.get('isSubmittable', '')}")

    pkg = item.get("packageInformation") or {}
    if pkg:
        parts.append(f"Package Name: {pkg.get('name', '')}")
        parts.append(f"Package Description: {pkg.get('description', '')}")
        parts.append(f"Procurement Method: {pkg.get('procurement_method', '')}")
        parts.append(f"Procurement Type: {pkg.get('procurement_type', '')}")
        parts.append(f"Award Type: {pkg.get('award_type', '')}")
        parts.append(f"Bid Security Amount: {pkg.get('bid_security_amount', '')}")
        parts.append(f"Bid Security Form: {pkg.get('bid_security_form', '')}")
        parts.append(f"Package Submission Deadline: {pkg.get('submission_deadline', '')}")
        parts.append(f"Clarification Deadline: {pkg.get('clarification_deadline', '')}")
        addr = pkg.get("address") or {}
        if addr:
            parts.append(f"Location: {addr.get('town', '')}, {addr.get('street', '')}")

    return "\n".join(p for p in parts if p and not p.endswith(": "))


def _egp_worker(result_queue):
    local_candidates = []
    try:
        found = scrape_egp(local_candidates)
        result_queue.put({"error": None, "candidates": local_candidates, "found": found})
    except Exception as e:
        result_queue.put({"error": str(e), "candidates": local_candidates, "found": 0})


def run_egp_with_timeout(candidates: list, timeout_seconds: int = 600) -> int:
    """Runs the eGP engine in its own process with a hard wall-clock timeout."""
    ctx = multiprocessing.get_context("spawn")
    result_queue = ctx.Queue()
    proc = ctx.Process(target=_egp_worker, args=(result_queue,))
    proc.start()
    proc.join(timeout=timeout_seconds)

    if proc.is_alive():
        print(f"⏱️ eGP engine exceeded {timeout_seconds}s and appears hung — terminating it so the rest of the scan cycle isn't blocked. Will retry next cycle.", flush=True)
        proc.terminate()
        proc.join(timeout=10)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return 0

    try:
        result = result_queue.get_nowait()
    except Exception:
        print("⚠️ eGP worker process exited without returning a result (likely crashed before reporting back).", flush=True)
        return 0

    if result["error"]:
        print(f"❌ eGP engine failed in subprocess: {result['error']}", flush=True)

    candidates.extend(result["candidates"])
    return result["found"]


# ==================== ENGINE: eGP (API — no browser) ====================
def scrape_egp(candidates: list):
    """
    Scrape eGP via the public /po-gw/cms-v2/api/sourcing/get-sourcing API.

    Playwright is intentionally NOT used here: the site detects CDP/automation
    and redirects every browser session to /egp/inspect-not-allowed. The
    sourcing API is public (no login required) and already returns title,
    reference, closing date, bid security, description, and procuring entity.
    """
    print(f"[{datetime.now()}] 🔍 Running eGP Engine (API)...", flush=True)
    found = 0
    seen_this_scan = set()

    # How many results to pull per search term
    TOP_PER_TERM = int(os.environ.get("EGP_TOP_PER_TERM", "25"))

    try:
        for term in EGP_SEARCH_TERMS:
            try:
                # API supports ?search= and ?top=
                from urllib.parse import quote
                q = quote(term)
                data = _egp_api_get(f"/cms-v2/api/sourcing/get-sourcing?search={q}&top={TOP_PER_TERM}")
                if not data or "items" not in data:
                    print(f"eGP search '{term}': no data", flush=True)
                    continue

                items = data.get("items") or []
                total = data.get("total", len(items))
                print(f"eGP search '{term}': {len(items)} items (total matching ~{total})", flush=True)

                for item in items:
                    try:
                        title_text = (item.get("lotName") or "").strip()
                        ref_no = (item.get("procurementReferenceNo") or item.get("lotReferenceNo") or "").strip()
                        if not title_text:
                            continue

                        row_key = ref_no or item.get("id") or title_text
                        if row_key in seen_this_scan:
                            continue
                        seen_this_scan.add(row_key)

                        if not is_relevant_tender(title_text):
                            continue

                        # Only alert on tenders that are still open for submission.
                        # The public API returns years of historical rows; without this
                        # filter, WhatsApp links often land on closed/expired packages.
                        is_submittable = bool(item.get("isSubmittable"))
                        deadline = (
                            item.get("submissionDeadline")
                            or (item.get("packageInformation") or {}).get("submission_deadline")
                            or ""
                        )
                        deadline_ok = False
                        if deadline:
                            try:
                                dl = datetime.fromisoformat(str(deadline).replace("Z", "+00:00"))
                                now = datetime.now(timezone.utc)
                                if dl.tzinfo is None:
                                    dl = dl.replace(tzinfo=timezone(timedelta(hours=3)))
                                # require at least ~1 hour remaining
                                deadline_ok = dl > now + timedelta(hours=1)
                            except Exception:
                                deadline_ok = False

                        # DEBUG: this item cleared the keyword filter — log exactly
                        # why it's kept or dropped, so a suspiciously-empty scan can
                        # be diagnosed (genuinely-closed tenders vs. a filter bug)
                        # instead of guessed at. Set EGP_DEBUG=0 to silence this.
                        if os.environ.get("EGP_DEBUG", "1").strip().lower() not in ("0", "false", "no"):
                            print(
                                f"🔬 eGP kw-match: {title_text[:70]} | ref={ref_no or item.get('id')} | "
                                f"isSubmittable={is_submittable} | deadline={deadline or '(none)'} | "
                                f"deadline_ok={deadline_ok}",
                                flush=True,
                            )

                        if not is_submittable and not deadline_ok:
                            continue
                        if deadline and not deadline_ok:
                            continue  # known past deadline
                        if not is_submittable:
                            # no usable deadline either — skip closed rows
                            continue

                        tender_id = f"egp_{ref_no or item.get('id') or title_text}"
                        if is_already_notified(tender_id):
                            continue

                        # NOTE: do NOT mark_as_notified() here. Marking happens
                        # only after the tender is actually resolved — either
                        # genuinely rejected by the AI or actually sent via
                        # WhatsApp — back in check_for_tenders(). Marking this
                        # early meant a transient AI/batch failure permanently
                        # buried a candidate that was never shown to anyone.
                        found += 1

                        detail_text = _egp_build_detail_text(item)

                        # SPA deep-link (from Angular router.navigate):
                        # /egp/bids/all/{tendering|purchasing|...}/{id}/open
                        # Purchasing uses sourceId; Tendering uses lotInPackageId || lotId
                        src_app = (item.get("sourceApplication") or "").strip()
                        app_path = {
                            "Tendering": "tendering",
                            "Auctioning": "auctioning",
                            "Purchasing": "purchasing",
                            "Prequalification": "prequalification",
                        }.get(src_app, "tendering")
                        if src_app == "Purchasing":
                            detail_id = (
                                item.get("sourceId")
                                or item.get("lotInPackageId")
                                or item.get("lotId")
                                or item.get("id")
                                or ""
                            )
                        else:
                            detail_id = (
                                item.get("lotInPackageId")
                                or item.get("lotId")
                                or item.get("id")
                                or ""
                            )
                        # Only link open packages we already filtered for
                        if detail_id:
                            detail_link = f"{EGP_BASE}/egp/bids/all/{app_path}/{detail_id}/open"
                        else:
                            detail_link = f"{EGP_BASE}/egp/bids/all"

                        print(
                            f"📄 eGP candidate: {title_text[:70]} | deadline={deadline} | "
                            f"submittable={is_submittable} | link_id={detail_id[:8]}…",
                            flush=True,
                        )

                        candidates.append({
                            "id": tender_id,
                            "source": "egp",
                            "title": title_text,
                            "ref_no": ref_no,
                            "detail_text": detail_text[:6000],
                            "link": detail_link,
                            "deadline": deadline,  # API closing date for WhatsApp fallback
                        })
                    except Exception:
                        continue

                time.sleep(0.5)  # be polite between search terms

            except Exception as e:
                print(f"⚠️ eGP search for '{term}' failed: {e}", flush=True)
                continue

        print(f"eGP engine complete. Collected {found} candidate matches for AI review.", flush=True)
        return found

    except Exception as e:
        print(f"❌ eGP extraction engine down: {e}", flush=True)
        return 0


def check_for_tenders():
    print("=================== STARTING SCAN CYCLE ===================", flush=True)

    candidates = []  # populated by both scrapers; analyzed in ONE batch AI call below

    try:
        merkato_found = scrape_2merkato(candidates)
    except Exception:
        print("❌ 2merkato engine crashed with an unhandled exception:", flush=True)
        print(traceback.format_exc(), flush=True)
        merkato_found = 0

    try:
        # API-based eGP does not hang like the old Playwright path, so run
        # in-process to guarantee candidates are returned (subprocess timeout
        # previously killed a successful 182-item collection before results
        # could be passed back to the parent).
        egp_found = scrape_egp(candidates)
    except Exception:
        print("❌ eGP engine crashed with an unhandled exception:", flush=True)
        print(traceback.format_exc(), flush=True)
        egp_found = 0

    print(f"🧮 {len(candidates)} total candidate tenders collected ({merkato_found} 2merkato, {egp_found} eGP) — running AI review (Groq×2: merkato + eGP)", flush=True)

    total_sent = 0
    if candidates:
        # ONE (or a handful of) Groq call(s) for the entire scan cycle, covering
        # every candidate from both engines — instead of the old per-tender
        # approach that could burn a day's free-tier quota in a single scan.
        # NOTE: this only batches the AI ANALYSIS step, not the WhatsApp
        # notifications — each tender still gets sent as its own message
        # below, since that's easier to read/forward than one long combined
        # message.
        batch_results = batch_analyze_tenders(candidates)

        for i, c in enumerate(candidates):
            result = batch_results.get(i) if batch_results is not None else None

            if batch_results is None:
                # The whole batch call failed even after retries (e.g. quota
                # exhausted). Fall back to sending everything that passed the
                # keyword filter, clearly flagged as unverified, rather than
                # silently dropping legitimate tenders.
                is_relevant = True
                match_score = "?"
                reason = "AI batch analysis unavailable — keyword match only"
                procurement_object = ""
                constraints = "Unknown — AI unavailable"
                closing_date = (c.get("deadline") or "").strip() or "Unknown — AI unavailable"
                ai_verified = False
            elif result is None:
                # Batch call succeeded but this particular index was missing
                # from the response — same fallback, just for one item
                is_relevant = True
                match_score = "?"
                reason = "Missing from AI batch response — keyword match only"
                procurement_object = ""
                constraints = "Unknown — AI unavailable"
                closing_date = (c.get("deadline") or "").strip() or "Unknown — AI unavailable"
                ai_verified = False
            else:
                is_relevant = bool(result.get("relevant", True))
                match_score = result.get("match_score", "?")
                reason = result.get("reason", "")
                procurement_object = result.get("procurement_object", "")
                # "Not stated in provided text" (not "None identified") is the
                # correct fallback here — the AI was only shown an excerpt of
                # the page, so an empty result means "not found in what we
                # sent it", not "confirmed there are none".
                constraints = result.get("constraints") or "Not stated in provided text"
                closing_date = result.get("closing_date") or "Not stated in provided text"
                ai_verified = True

            # Prefer real API deadline when AI left it unknown / not stated
            api_deadline = (c.get("deadline") or "").strip()
            if api_deadline and (
                not closing_date
                or closing_date in (
                    "Unknown — AI unavailable",
                    "Not stated in provided text",
                    "?",
                )
            ):
                closing_date = api_deadline

            if not is_relevant:
                # Log the AI's actual reasoning, not just the title — makes
                # it possible to sanity-check a borderline rejection (e.g. a
                # title that LOOKED relevant) without digging through the
                # WhatsApp thread for context that was never sent there
                print(f"🤖 AI rejected as not relevant, skipping alert: {c['title']} — reason: {reason or '(no reason returned)'}", flush=True)
                # This is a genuine, AI-verified rejection (not a transient
                # failure — those keep is_relevant forced True above), so it's
                # safe to permanently mark it and never re-surface it.
                mark_as_notified(c["id"])
                continue

            # When AI fails for an item, still alert if the title has a STRONG
            # keyword (dialysis, ultrasound equipment, etc.). Only skip weak
            # generic matches. Set SKIP_UNVERIFIED_ALERTS=1 to suppress all unverified.
            skip_all_unverified = os.environ.get("SKIP_UNVERIFIED_ALERTS", "0").strip().lower() in ("1", "true", "yes")
            if not ai_verified:
                title_l = c["title"].lower()
                strong_hit = any(
                    k in title_l
                    for k in (
                        "dialysis", "hemodialysis", "hemodial", "bbraun", "b braun",
                        "dialog+", "ultrasound", "x-ray", "xray", "ct scan",
                        "steriliz", "autoclave", "cssd", "reverse osmosis", "ro system",
                        "water treatment", "biomedical", "medical imaging",
                        "maintenance and repair", "calibration",
                    )
                )
                if skip_all_unverified or not strong_hit:
                    print(f"⏭️ Skipping unverified alert: {c['title'][:60]}", flush=True)
                    continue
                print(f"⚠️ AI unavailable but strong keyword match — still alerting: {c['title'][:60]}", flush=True)

            def _short(s, n=100):
                s = (s or "").strip()
                return s if len(s) <= n else s[: n - 1] + "…"

            reason_s = _short(str(reason), 80)
            constraints_s = _short(str(constraints), 100)
            object_s = _short(str(procurement_object), 80)
            title_s = _short(c["title"], 120)

            source_label = "eGP" if c["source"] == "egp" else "2merkato"
            lines = [f"🔔 *New tender ({source_label})*"]
            if not ai_verified:
                lines.append("⚠️ *Unverified* (AI limited — keyword match)")
            lines.append(f"📋 {title_s}")
            if c.get("ref_no"):
                lines.append(f"📄 Ref: {c['ref_no']}")
            if object_s:
                lines.append(f"🏷️ {object_s}")
            lines.append(f"🎯 {match_score}% — {reason_s}")
            if constraints_s and constraints_s not in (
                "Not stated in provided text",
                "Unknown — AI unavailable",
            ):
                lines.append(f"⚠️ {constraints_s}")
            lines.append(f"📅 Close: {closing_date}")
            # Put the URL alone on its own line — WhatsApp often fails to make
            # links tappable when they sit on the same line as *markdown* or emoji.
            link = (c.get("link") or "").strip()
            if link:
                lines.append("")
                lines.append(link)

            alert = "\n".join(lines)
            send_whatsapp(alert)
            total_sent += 1
            # Only now — after the alert has actually gone out — is it safe
            # to permanently mark this tender as notified.
            mark_as_notified(c["id"])
            time.sleep(2)

    total = total_sent
    print(f"=================== SCAN COMPLETE: {total} ALERTS SENT ({len(candidates)} candidates reviewed) ===================", flush=True)

    if total == 0:
        send_whatsapp("🔍 *Tender Monitor Scan Completed.*\nNo new unique medical equipment or maintenance matches found on 2merkato or eGP.")


def monitoring_loop():
    # Configurable via Render env var so this doesn't need a code change next time —
    # defaults to 6 hours if SCAN_INTERVAL_HOURS isn't set
    try:
        interval_hours = float(os.environ.get("SCAN_INTERVAL_HOURS", "6"))
    except ValueError:
        interval_hours = 6
    while True:
        try:
            check_for_tenders()
        except Exception:
            print("❌ check_for_tenders crashed at the top level:", flush=True)
            print(traceback.format_exc(), flush=True)
        time.sleep(interval_hours * 3600)


# ==================== FLASK ROUTES ====================
@app.route('/')
def home():
    return "Medical Tender Tracking Service Is Online!", 200


@app.route('/test-check')
def manual_test():
    # SECURITY: without this check, anyone who finds the public Render URL
    # could trigger a scan on demand — burning Groq API quota and sending
    # WhatsApp messages at will. Require a shared secret token to proceed.
    if not TEST_CHECK_TOKEN:
        return "Manual trigger is disabled: TEST_CHECK_TOKEN is not configured.", 503

    provided_token = request.args.get("token", "")
    if not hmac.compare_digest(provided_token, TEST_CHECK_TOKEN):
        return "Unauthorized", 401

    threading.Thread(target=check_for_tenders).start()
    return "Scraper sync cycle triggered!", 200


# Ensure the dedup table exists as soon as the module loads (covers both
# `python app.py` local runs and gunicorn/production imports)
init_db()

if __name__ == "__main__":
    threading.Thread(target=monitoring_loop, daemon=True).start()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
