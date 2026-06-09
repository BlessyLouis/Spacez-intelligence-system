"""
Spacez · AI Review Intelligence System
────────────────────────────────────────
Architecture (Req 8):
  90 %  Pure Python / Pandas  — column detection, normalisation, noise filter,
                                keyword classification, pattern detection,
                                priority scoring, confidence, business metrics
  10 %  Gemini (one batch)    — root cause + action per issue cluster
                              — executive summary
                              — copilot Q&A
"""

import streamlit as st
import pandas as pd
import json
import re
import hashlib
import plotly.graph_objects as go
import google.generativeai as genai

# ═══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Spacez · Review Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif}
.main{background:#F7F5F0}
.block-container{padding-top:1.5rem;padding-bottom:2rem}

.metric-card{background:white;border-radius:12px;padding:1.2rem 1.4rem;border:1px solid #E8E4DC;margin-bottom:.8rem}
.metric-card .label{font-size:.72rem;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:#8B8578;margin-bottom:.3rem}
.metric-card .value{font-family:'Sora',sans-serif;font-size:2rem;font-weight:700;color:#1A1915;line-height:1}
.metric-card .delta{font-size:.8rem;margin-top:.3rem;color:#8B8578}

.issue-card{background:white;border-radius:12px;padding:1.4rem;border:1px solid #E8E4DC;margin-bottom:1rem;border-left:4px solid #E8E4DC}
.issue-card.high{border-left-color:#D94F3D}
.issue-card.medium{border-left-color:#E8971A}
.issue-card.low{border-left-color:#3B8A5C}

.issue-title{font-family:'Sora',sans-serif;font-size:1rem;font-weight:700;color:#1A1915;margin-bottom:.3rem}
.issue-meta{font-size:.78rem;color:#8B8578;margin-bottom:.8rem}

.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.7rem;font-weight:600;margin-right:6px;letter-spacing:.04em}
.badge-high{background:#FDECEA;color:#9B2218}
.badge-medium{background:#FEF3DE;color:#8A5310}
.badge-low{background:#E6F4EC;color:#1E6641}
.badge-ops{background:#EEF2FF;color:#3730A3}
.badge-biz{background:#F0FDF4;color:#14532D}
.badge-care{background:#FFF7ED;color:#7C2D12}
.badge-person{background:#FDF4FF;color:#6B21A8}

.evidence-box{background:#F7F5F0;border-radius:8px;padding:.8rem 1rem;margin-top:.6rem;border-left:3px solid #C8C4BC}
.evidence-quote{font-size:.82rem;color:#4A4740;font-style:italic;margin-bottom:.4rem;padding-left:.5rem;border-left:2px solid #C8C4BC}
.section-label{font-size:.7rem;font-weight:600;color:#8B8578;text-transform:uppercase;letter-spacing:.06em;margin-bottom:.4rem}

.explain-row{display:flex;gap:.6rem;align-items:flex-start;margin-bottom:.4rem;font-size:.8rem}
.explain-step{min-width:80px;font-weight:600;color:#6B6560}
.explain-text{color:#3A3730}

.step-bar{background:#F0EDE6;border-radius:8px;padding:.6rem 1rem;margin-bottom:.4rem;font-size:.82rem;color:#3A3730}
.step-bar.active{background:#1A1915;color:#F7F5F0;font-weight:600}
.step-bar.done{background:#E6F4EC;color:#1E6641}

[data-testid="stSidebar"]{background:#1A1915}
[data-testid="stSidebar"] *{color:#E8E4DC !important}
[data-testid="stSidebar"] .stTextInput input{background:#2C2A25;border-color:#3A3730;color:#E8E4DC}
[data-testid="stSidebar"] hr{border-color:#3A3730}

h1,h2,h3{font-family:'Sora',sans-serif;font-weight:700;color:#1A1915}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════
SEVERITY_SCORE = {"High": 3, "Medium": 2, "Low": 1}
OWNER_URGENCY  = {"Operations": 1.2, "Caretaker": 1.1, "Business": 1.0, "Vendor": 0.9, "Unknown": 0.8}

# ── Req 1: Deterministic category keyword map ─────────────────
# Each entry: (category, owner, severity, caretaker_controllable, keywords)
CATEGORY_RULES = [
    ("Maintenance",     "Operations",  "High",   False, [
        "heating", "heater", "boiler", "hot water", "broken", "not working",
        "malfunction", "repair", "leak", "pipe", "electrical", "power cut",
        "ac", "air conditioning", "fan", "pump", "equipment"
    ]),
    ("Cleanliness",     "Caretaker",   "High",   True,  [
        "pool", "dirty pool", "cloudy water", "leaves", "algae", "green water",
        "pool maintenance", "pool cleaning", "pool cleanliness"
    ]),
    ("Housekeeping",    "Caretaker",   "Medium", True,  [
        "clean", "cleanliness", "dust", "dusty", "dirty", "hygiene", "stain",
        "hair", "crumbs", "grime", "grimy", "floors", "bathroom", "housekeeping",
        "swept", "mopped", "smells", "odour", "bins", "garbage", "trash"
    ]),
    ("Check-in",        "Caretaker",   "High",   True,  [
        "check-in", "check in", "checkin", "arrival", "waited", "waiting",
        "late", "delay", "delayed", "no one", "nobody", "unreachable",
        "not reachable", "not respond", "no response", "outside", "gate"
    ]),
    ("Communication",   "Caretaker",   "Medium", True,  [
        "response", "respond", "communication", "reply", "contact", "reach",
        "message", "instructions", "no answer", "unresponsive", "slow reply",
        "not informed", "no information", "unclear"
    ]),
    ("Amenities",       "Operations",  "Medium", False, [
        "bbq", "jacuzzi", "pool table", "gym", "equipment", "amenity",
        "amenities", "not available", "out of service", "broken amenity",
        "missing", "listed", "advertised"
    ]),
    ("Safety",          "Operations",  "High",   False, [
        "safety", "unsafe", "dangerous", "hazard", "slippery", "injury",
        "accident", "security", "lock", "fence", "gate broken"
    ]),
    ("Staff Behaviour", "Caretaker",   "High",   True,  [
        "rude", "unfriendly", "unhelpful", "attitude", "behaviour", "behavior",
        "staff", "caretaker", "host", "impolite", "unprofessional"
    ]),
    # Non-controllable / Business issues
    ("Amenities",       "Business",    "Low",    False, [
        "wifi", "wi-fi", "internet", "connection", "network", "signal",
        "connectivity", "broadband", "slow internet", "no wifi"
    ]),
    ("Other",           "Business",    "Low",    False, [
        "photos", "pictures", "listing", "different", "smaller", "misleading",
        "not as described", "inaccurate", "road", "location", "distance",
        "price", "pricing", "expensive", "value"
    ]),
]

# Non-controllable keyword set for quick lookup
NON_CONTROLLABLE_KEYWORDS = {
    "wifi","wi-fi","internet","connection","network","signal","connectivity",
    "photos","pictures","listing","misleading","road","location","distance",
    "price","pricing","expensive","infrastructure"
}


# ═══════════════════════════════════════════════════════════════
# GEMINI HELPERS  (Req 2 + 3 + 4)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource
def get_model(api_key: str):
    """Cache the Gemini model so it's only initialised once (Req 4)."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def gemini_call(model, prompt: str, max_tokens: int = 3000) -> str:
    """Single Gemini call with retry on rate-limit."""
    import time
    last_err = None
    for attempt in range(4):
        try:
            resp = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=max_tokens, temperature=0.2
                ),
            )
            return resp.text.strip()
        except Exception as e:
            err = str(e)
            if "API_KEY_INVALID" in err or ("403" in err and "quota" not in err.lower()):
                st.error(f"Gemini API key rejected (403). Check your GEMINI_API_KEY secret.\n\n{err}")
                st.stop()
            if "429" in err or "quota" in err.lower() or "rate" in err.lower():
                wait = 20 * (attempt + 1)
                time.sleep(wait)
                last_err = e
                continue
            raise
    raise last_err


# ── Req 2 + 3: Single batch prompt for ALL clusters ───────────
BATCH_INSIGHTS_PROMPT = """You are an operations analyst for Spacez, a luxury villa company.

For each issue cluster below, write:
1. root_cause   — 1-2 sentences citing exact numbers and names from the data
2. action       — 2-3 concrete sentences: who does what, by when, measurable outcome

Issue clusters:
{clusters_block}

Reply ONLY as a JSON array (same order as clusters), each object:
{{"root_cause": "...", "action": "..."}}
No markdown, no preamble."""

EXECUTIVE_SUMMARY_PROMPT = """You are a senior hospitality consultant reviewing Spacez villa performance.

Portfolio data:
{summary}

Write a concise executive summary (4-6 sentences) covering:
- Overall portfolio health
- Top 2-3 risks requiring immediate action
- Recommended priorities for this week
- One positive finding

Plain text only. No headers, no bullets."""

@st.cache_data(show_spinner=False)
def run_gemini_batch(_model_key: str, clusters_json: str) -> tuple[list, list, str]:
    """
    ONE Gemini call: root causes + actions + executive summary combined.
    Cached by api_key + clusters_json — same data never re-triggers Gemini.
    """
    model    = get_model(_model_key)
    clusters = json.loads(clusters_json)

    # Build compact cluster block
    block = ""
    for i, c in enumerate(clusters):
        block += (
            f"[{i+1}] {c['category']} at {c['property']} | "
            f"Freq:{c['frequency']} | {c.get('priority','?')} priority | "
            f"Owner:{c['owner']} | Caretaker:{c['caretaker']} | "
            f"Recurring:{c['is_recurring']}\n"
            f"Reviews: {'; '.join(c.get('descriptions',[])[:2])}\n"
        )

    high_count = sum(1 for c in clusters if c.get("priority") == "High")
    props      = list({c["property"] for c in clusters})
    avg_rating = round(
        sum(c.get("avg_rating_normalised", 0.5) for c in clusters) / max(len(clusters), 1) * 5, 1
    )

    combined_prompt = f"""You are an operations analyst for Spacez luxury villas.

Issue clusters from guest reviews:
{block.strip()}

Portfolio: {len(clusters)} clusters across {len(props)} properties, avg rating {avg_rating}/5, {high_count} high-priority.

Return ONLY this JSON (no markdown, no extra text):
{{
  "clusters": [
    {{"root_cause": "1-2 sentences with specific numbers and names", "action": "concrete: who does what by when"}}
  ],
  "exec_summary": "4-5 sentences: portfolio health, top risks, this week priorities"
}}
The clusters array must have exactly {len(clusters)} items in the same order as input."""

    root_causes, actions, exec_summary = [], [], ""

    try:
        raw = gemini_call(model, combined_prompt, max_tokens=2500)
        raw = re.sub(r"```json|```", "", raw).strip()
        brace_start = raw.find("{")
        brace_end   = raw.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            raw = raw[brace_start:brace_end]
        data = json.loads(raw)
        for r in data.get("clusters", []):
            root_causes.append((r.get("root_cause") or "").strip() or None)
            actions.append((r.get("action") or "").strip() or None)
        exec_summary = (data.get("exec_summary") or "").strip()
    except Exception:
        pass  # fall through to deterministic defaults below

    # Deterministic fallbacks — never show empty or keyword strings
    while len(root_causes) < len(clusters):
        root_causes.append(None)
    while len(actions) < len(clusters):
        actions.append(None)

    for i, c in enumerate(clusters):
        if not root_causes[i]:
            root_causes[i] = (
                f"{c['frequency']} guest review(s) at {c['property']} flagged "
                f"{c['category'].lower()} as an issue. Assigned caretaker: {c['caretaker']}."
            )
        if not actions[i]:
            actions[i] = (
                f"{c['owner']} team to inspect {c['category'].lower()} at {c['property']} "
                f"within 48 hours, brief caretaker {c['caretaker']} on corrective steps, "
                f"and confirm resolution in the ops log."
            )

    if not exec_summary or len(exec_summary) < 40:
        top = clusters[0] if clusters else {}
        exec_summary = (
            f"Portfolio analysis across {len(props)} propert{'y' if len(props)==1 else 'ies'} "
            f"found {len(clusters)} issue cluster{'s' if len(clusters)!=1 else ''} "
            f"(avg rating {avg_rating}/5, {high_count} high priority). "
            f"Most frequent: {top.get('category','—')} at {top.get('property','—')} "
            f"({top.get('frequency',0)} mentions, caretaker {top.get('caretaker','—')}). "
            f"Immediate action required on all high-priority caretaker-controllable clusters."
        )

    return root_causes[:len(clusters)], actions[:len(clusters)], exec_summary




# ═══════════════════════════════════════════════════════════════
# STEP 1: LOAD & NORMALISE  (Pandas only, Req 6)
# ═══════════════════════════════════════════════════════════════

def load_reviews(file_bytes: bytes) -> pd.DataFrame:
    import io
    df = pd.read_excel(io.BytesIO(file_bytes))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def normalise_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Booking.com /10 → /1, others /5 → /1. Preserves Req 7."""
    platform_col = next((c for c in df.columns if "platform" in c or "source" in c), None)
    rating_col   = next((c for c in df.columns if "rating" in c or "score" in c), None)
    if not rating_col:
        df["normalised_rating"] = 0.7
        return df

    def _norm(row):
        plat   = str(row.get(platform_col, "")).lower() if platform_col else ""
        rating = row.get(rating_col)
        if pd.isna(rating):
            return None
        return (float(rating) / 10.0) if "booking" in plat else (float(rating) / 5.0)

    df["normalised_rating"] = df.apply(_norm, axis=1)
    return df


def detect_columns(df: pd.DataFrame):
    cols = df.columns.tolist()
    REVIEW_KEYWORDS   = ["review_text","comment","feedback","review_body","text","review"]
    review_col    = next((c for c in cols if any(k in c for k in REVIEW_KEYWORDS) and "id" not in c), cols[0])
    property_col  = next((c for c in cols if "property" in c), None)
    caretaker_col = next((c for c in cols if "caretaker" in c or "host" in c), None)
    return review_col, property_col, caretaker_col


def filter_noise(df: pd.DataFrame, review_col: str) -> pd.DataFrame:
    NOISE = re.compile(
        r"^(amazing|perfect|great|loved it|excellent|fantastic|wonderful|5 stars|good)[!.]*$", re.I
    )
    col = df[review_col].astype(str).replace("nan", "").str.strip()
    df["is_noise"] = (col.str.len() < 15) | col.str.match(NOISE)
    return df


# ═══════════════════════════════════════════════════════════════
# STEP 2: DETERMINISTIC KEYWORD CLASSIFICATION  (Req 1)
# ═══════════════════════════════════════════════════════════════

def _build_compiled_rules():
    """Pre-compile all keyword patterns once at import time (Req 6)."""
    compiled = []
    for (category, owner, severity, controllable, keywords) in CATEGORY_RULES:
        pattern = re.compile(
            r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b",
            re.IGNORECASE
        )
        compiled.append((category, owner, severity, controllable, pattern, keywords))
    return compiled

COMPILED_RULES = _build_compiled_rules()


def classify_review(text: str) -> list[dict]:
    """
    Pure Python keyword classifier. Returns a list of issue dicts.
    No API calls. O(n_rules × len_text). (Req 1)
    """
    text_lower = text.lower()
    matched = []
    seen_categories = set()

    for (category, owner, severity, controllable, pattern, keywords) in COMPILED_RULES:
        if pattern.search(text_lower):
            if category in seen_categories:
                continue  # avoid duplicate categories from overlapping rules
            seen_categories.add(category)

            # Detect non-controllable via keyword overlap
            matched_keywords = [k for k in keywords if k in text_lower]
            is_non_ctrl = any(k in NON_CONTROLLABLE_KEYWORDS for k in matched_keywords)
            if is_non_ctrl:
                controllable = False
                owner = "Business" if owner == "Caretaker" else owner

            matched.append({
                "category":               category,
                "owner":                  owner,
                "severity":               severity,
                "caretaker_controllable": controllable,
                "non_controllable_reason": ", ".join(k for k in matched_keywords if k in NON_CONTROLLABLE_KEYWORDS),
                "description":            f"{category} issue mentioned in review",
                "likely_cause":           f"Keyword match: {', '.join(matched_keywords[:3])}",
                "matched_keywords":       matched_keywords[:5],
            })

    return matched


def classify_all_reviews(df: pd.DataFrame, review_col: str,
                          property_col: str, caretaker_col: str) -> list[dict]:
    """
    Vectorised keyword classification over entire dataframe. (Req 1, Req 6)
    Returns flat list of issue records, one per (review, matched_category).
    """
    records = []
    for _, row in df.iterrows():
        text     = str(row.get(review_col, ""))
        prop     = str(row.get(property_col, "Unknown")) if property_col else "Unknown"
        caretaker= str(row.get(caretaker_col, "Unknown")) if caretaker_col else "Unknown"
        rating   = float(row.get("normalised_rating", 0.5))
        review_t = text[:300]

        issues = classify_review(text)
        if issues:
            for issue in issues:
                records.append({
                    **issue,
                    "property":          prop,
                    "caretaker":         caretaker,
                    "normalised_rating": rating,
                    "review_text":       review_t,
                })
        # Even non-matching reviews contribute to evidence pool
    return records


# ═══════════════════════════════════════════════════════════════
# STEP 3: PATTERN DETECTION  (Pure Pandas, Req 6, Req 7)
# ═══════════════════════════════════════════════════════════════

def detect_patterns(records: list[dict]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Frequency per (property, category)
    freq = (df.groupby(["property","category"])
              .size()
              .reset_index(name="frequency"))
    freq["is_recurring"] = freq["frequency"] >= 3
    df = df.merge(freq, on=["property","category"])

    # Average rating per cluster
    rating_avg = (df.groupby(["property","category"])["normalised_rating"]
                    .mean()
                    .reset_index(name="avg_rating_normalised"))
    df = df.merge(rating_avg, on=["property","category"])

    # Person-level pattern: same caretaker, same issue, 2+ properties
    cross = (
        df[df["caretaker_controllable"] == True]
        .groupby(["caretaker","category"])["property"]
        .nunique()
        .reset_index(name="property_count")
    )
    cross["person_level_pattern"] = cross["property_count"] >= 2
    df = df.merge(
        cross[["caretaker","category","person_level_pattern"]],
        on=["caretaker","category"], how="left"
    )
    df["person_level_pattern"] = df["person_level_pattern"].fillna(False)

    # Evidence: best 3 quotes per cluster (shortest = most specific)
    evidence = (df.groupby(["property","category"])["review_text"]
                  .apply(lambda x: sorted(x.dropna().tolist(), key=len)[:3])
                  .reset_index(name="evidence_reviews"))
    df = df.merge(evidence, on=["property","category"])

    # Descriptions list per cluster
    desc = (df.groupby(["property","category"])["description"]
              .apply(list)
              .reset_index(name="descriptions"))
    df = df.merge(desc, on=["property","category"])

    return df


# ═══════════════════════════════════════════════════════════════
# STEP 4: PRIORITY + CONFIDENCE  (Pure Python, Req 6, Req 7)
# ═══════════════════════════════════════════════════════════════

def compute_priority_score(row: pd.Series, portfolio_avg: float) -> float:
    freq_s    = min(row["frequency"] / 3, 3.0)
    sev_s     = SEVERITY_SCORE.get(row["severity"], 1)
    owner_m   = OWNER_URGENCY.get(row["owner"], 1.0)
    recur_m   = 1.5 if row["is_recurring"] else 1.0
    person_m  = 1.3 if row.get("person_level_pattern", False) else 1.0
    impact    = max(0, portfolio_avg - row["avg_rating_normalised"])
    rating_m  = 1.0 + (impact * 2)
    return round(freq_s * sev_s * owner_m * recur_m * person_m * rating_m, 2)


def priority_label(score: float) -> str:
    if score >= 10: return "High"
    if score >= 4:  return "Medium"
    return "Low"


def compute_confidence(row: pd.Series, total_reviews: int) -> int:
    freq_sig  = min(row["frequency"] / max(total_reviews * 0.08, 1), 1.0) * 55
    rec_bonus = 20 if row["is_recurring"] else 0
    per_bonus = 15 if row.get("person_level_pattern", False) else 0
    sev_bonus = {"High": 8, "Medium": 4, "Low": 0}.get(row["severity"], 0)
    ev_bonus  = min(len(row.get("evidence_reviews", [])) * 3, 9)
    return min(int(freq_sig + rec_bonus + per_bonus + sev_bonus + ev_bonus), 99)


# ═══════════════════════════════════════════════════════════════
# STEP 5: BUSINESS METRICS  (Pure Pandas, Req 7)
# ═══════════════════════════════════════════════════════════════

def compute_business_metrics(clusters: pd.DataFrame,
                              total_reviews: int,
                              portfolio_avg: float) -> dict:
    high   = (clusters["priority"] == "High").sum()
    med    = (clusters["priority"] == "Medium").sum()
    total  = len(clusters)
    recur  = int(clusters["is_recurring"].sum())
    person = int(clusters["person_level_pattern"].sum())
    ctrl_pct = clusters["caretaker_controllable"].mean() if total else 0

    # Property health score (vectorised)
    penalty_map = {"High": 15, "Medium": 7, "Low": 2}
    clusters["_penalty"] = clusters.apply(
        lambda r: penalty_map.get(r["priority"], 0) * (1.5 if r["is_recurring"] else 1.0), axis=1
    )
    prop_scores = (100 - clusters.groupby("property")["_penalty"].sum()).clip(lower=0).round(1).to_dict()
    clusters.drop(columns=["_penalty"], inplace=True)

    ops_risk = min(100, round((high * 15 + med * 5 + recur * 8) / max(total, 1) * 10, 1))

    # Caretaker scores (vectorised)
    ct_stats = (clusters
                .groupby("caretaker")
                .apply(lambda g: pd.Series({
                    "controllable":     int(g["caretaker_controllable"].sum()),
                    "non_controllable": int((~g["caretaker_controllable"]).sum()),
                    "total":            len(g),
                    "penalty":          g[g["caretaker_controllable"]]
                                        .apply(lambda r: {"High":10,"Medium":5,"Low":1}.get(r["priority"],0), axis=1)
                                        .sum()
                }))
                .reset_index())
    caretaker_scores = {}
    for _, r in ct_stats.iterrows():
        pct   = r["controllable"] / max(r["total"], 1)
        score = max(0, round(100 - r["penalty"] * pct, 1))
        caretaker_scores[r["caretaker"]] = {
            "score": score,
            "controllable": int(r["controllable"]),
            "non_controllable": int(r["non_controllable"]),
            "total": int(r["total"]),
        }

    rating_impacts = (
        portfolio_avg - clusters.groupby("category")["avg_rating_normalised"].mean()
    ).round(3).to_dict()

    return {
        "total_issues":              total,
        "high_count":                int(high),
        "med_count":                 int(med),
        "recurring_count":           recur,
        "person_level_count":        person,
        "issue_resolution_potential": round(ctrl_pct * 100, 1),
        "property_health":           prop_scores,
        "ops_risk_score":            ops_risk,
        "caretaker_scores":          caretaker_scores,
        "rating_impacts":            rating_impacts,
    }


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE  (Req 4 — cached by file hash)
# ═══════════════════════════════════════════════════════════════

def _file_hash(file_bytes: bytes) -> str:
    return hashlib.md5(file_bytes).hexdigest()


@st.cache_data(show_spinner=False)
def run_full_pipeline(api_key: str, file_bytes: bytes, _file_hash: str):
    """
    _file_hash is the cache key — same file never reprocesses. (Req 4)
    90 % Pandas, 10 % Gemini (one batch call). (Req 6)
    """
    # ── Step 1: Load  ─────────────────────────────────────
    df = load_reviews(file_bytes)
    df = normalise_ratings(df)
    review_col, property_col, caretaker_col = detect_columns(df)
    df = filter_noise(df, review_col)

    total_reviews = len(df)
    portfolio_avg = df["normalised_rating"].dropna().mean()
    actionable    = df[~df["is_noise"]].reset_index(drop=True)

    # ── Step 2: Keyword classification (no Gemini)  ───────
    records = classify_all_reviews(actionable, review_col, property_col, caretaker_col)

    # ── Step 3: Pattern detection  ────────────────────────
    issues_df = detect_patterns(records)
    if issues_df.empty:
        return None, None, total_reviews, portfolio_avg, df, ""

    # ── Deduplicate to one row per cluster  ───────────────
    clusters = (
        issues_df
        .sort_values("frequency", ascending=False)
        .drop_duplicates(subset=["property","category"])
        .reset_index(drop=True)
    )

    # ── Step 4: Priority + confidence  ────────────────────
    clusters["rating_impact"]  = (portfolio_avg - clusters["avg_rating_normalised"]).round(3)
    clusters["priority_score"] = clusters.apply(lambda r: compute_priority_score(r, portfolio_avg), axis=1)
    clusters["priority"]       = clusters["priority_score"].apply(priority_label)
    clusters["confidence"]     = clusters.apply(lambda r: compute_confidence(r, total_reviews), axis=1)

    # ── Step 5 (Gemini): ONE batch call for all clusters  ─
    clusters_for_gemini = clusters[
        ["category","property","frequency","priority","owner",
         "caretaker","is_recurring","person_level_pattern",
         "descriptions","avg_rating_normalised","rating_impact","likely_cause"]
    ].to_dict("records")
    # Convert numpy booleans to Python booleans for JSON serialisation
    for c in clusters_for_gemini:
        c["is_recurring"]       = bool(c["is_recurring"])
        c["person_level_pattern"] = bool(c["person_level_pattern"])
        c["rating_impact"]      = float(c["rating_impact"])

    root_causes, actions, exec_summary = run_gemini_batch(
        api_key, json.dumps(clusters_for_gemini)
    )

    clusters["root_cause"]            = root_causes[:len(clusters)]
    clusters["action_recommendation"] = actions[:len(clusters)]

    # ── Step 6: Business metrics  ─────────────────────────
    metrics = compute_business_metrics(clusters, total_reviews, portfolio_avg)

    return clusters, metrics, total_reviews, portfolio_avg, df, exec_summary


# ═══════════════════════════════════════════════════════════════
# UI HELPERS
# ═══════════════════════════════════════════════════════════════

def priority_badge(p):
    cls = {"High":"badge-high","Medium":"badge-medium","Low":"badge-low"}.get(p,"badge-low")
    return f'<span class="badge {cls}">{p} priority</span>'

def owner_badge(o):
    cls = {"Operations":"badge-ops","Business":"badge-biz","Caretaker":"badge-care"}.get(o,"badge-low")
    return f'<span class="badge {cls}">{o}</span>'

def render_metric(label, value, delta=None):
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

def render_issue_card(row, port_avg5):
    p   = row["priority"]
    evs = row.get("evidence_reviews", [])

    ev_html = "".join(
        f'<div class="evidence-quote">"{str(ev).strip()[:180]}{"…" if len(str(ev).strip())>180 else ""}"</div>'
        for ev in evs[:3] if ev and len(str(ev).strip()) > 10
    )
    person_badge = (
        '<span class="badge badge-person">⚠ person-level pattern</span>'
        if row.get("person_level_pattern") else ""
    )
    rating5    = round(row["avg_rating_normalised"] * 5, 1)
    impact5    = round(row["rating_impact"] * 5, 1)
    impact_str = f"−{impact5}" if impact5 > 0 else f"+{abs(impact5)}"

    explain_html = f"""
    <div class="explain-row">
        <span class="explain-step">Detected:</span>
        <span class="explain-text">{row['frequency']} reviews matched keywords for {row['category']} at {row['property']}</span>
    </div>
    <div class="explain-row">
        <span class="explain-step">Priority:</span>
        <span class="explain-text">{p} — freq {row['frequency']} × severity {row['severity']} × rating impact {impact_str}/5</span>
    </div>
    <div class="explain-row">
        <span class="explain-step">Owner:</span>
        <span class="explain-text">{row['owner']} — {'caretaker-controllable' if row['caretaker_controllable'] else 'outside caretaker control'}</span>
    </div>
    <div class="explain-row">
        <span class="explain-step">Method:</span>
        <span class="explain-text">Keyword match: {', '.join(row.get('matched_keywords', [])[:4]) or 'pattern detection'}</span>
    </div>"""

    st.markdown(f"""
    <div class="issue-card {p.lower()}">
        <div class="issue-title">{row['category']} · {row['property']}</div>
        <div class="issue-meta">
            {priority_badge(p)} {owner_badge(row['owner'])} {person_badge}
            &nbsp;·&nbsp; {row['frequency']} reviews &nbsp;·&nbsp; Confidence {row['confidence']}%
            &nbsp;·&nbsp; Avg rating {rating5}/5 (portfolio {port_avg5}/5)
            &nbsp;·&nbsp; Rating impact <b>{impact_str}/5</b>
        </div>

        <div class="section-label">Root cause</div>
        <div style="font-size:.85rem;color:#3A3730;margin-bottom:.8rem">{row['root_cause']}</div>

        <div class="section-label">Evidence reviews ({min(len(evs),3)} shown)</div>
        <div class="evidence-box">{ev_html or '<span style="font-size:.8rem;color:#8B8578">No direct quotes stored.</span>'}</div>

        <div class="section-label" style="margin-top:.8rem">Recommended action</div>
        <div style="font-size:.85rem;color:#1A1915;font-weight:500;margin-bottom:.8rem">{row['action_recommendation']}</div>

        <div class="section-label">Explainability</div>
        {explain_html}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
try:
    api_key = st.secrets["GEMINI_API_KEY"]
except Exception:
    api_key = None

with st.sidebar:
    st.markdown("### 🏡 Spacez")
    st.markdown("**Review Intelligence System**")
    st.markdown("---")
    if not api_key:
        st.error("GEMINI_API_KEY not found in Streamlit secrets.")
    uploaded = st.file_uploader("Upload review Excel file", type=["xlsx","xls"])
    run_btn  = st.button("▶  Run analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("**Filters**")
    filter_priority = st.multiselect("Priority", ["High","Medium","Low"],  default=["High","Medium","Low"])
    filter_owner    = st.multiselect("Owner",
                                     ["Operations","Caretaker","Business","Vendor","Unknown"],
                                     default=["Operations","Caretaker","Business","Vendor","Unknown"])
    st.markdown("---")
    st.caption("Powered by Google Gemini 2.0 Flash")


# ═══════════════════════════════════════════════════════════════
# SESSION STATE
# ═══════════════════════════════════════════════════════════════
for k in ["clusters","metrics","total","port_avg","raw_df","exec_summary"]:
    if k not in st.session_state:
        st.session_state[k] = None


# ═══════════════════════════════════════════════════════════════
# RUN ANALYSIS  (Req 5 — step-by-step progress UX)
# ═══════════════════════════════════════════════════════════════
if run_btn:
    if not api_key:
        st.error("GEMINI_API_KEY missing. Add it in Streamlit Cloud → App settings → Secrets.")
        st.stop()
    if not uploaded:
        st.error("Please upload a review Excel file.")
        st.stop()

    file_bytes = uploaded.read()
    fhash      = _file_hash(file_bytes)

    # Step-by-step progress display (Req 5)
    progress_placeholder = st.empty()
    steps = [
        ("Loading & normalising reviews",     "🔄"),
        ("Detecting issues (keyword engine)", "🔍"),
        ("Finding patterns & clusters",       "📊"),
        ("Generating AI insights (Gemini)",   "🤖"),
        ("Building dashboard",                "✅"),
    ]

    def show_steps(active: int):
        html = ""
        for i, (label, icon) in enumerate(steps):
            if i < active:
                cls = "done";   marker = "✓"
            elif i == active:
                cls = "active"; marker = icon
            else:
                cls = "";       marker = "○"
            html += f'<div class="step-bar {cls}">{marker} Step {i+1}: {label}</div>'
        progress_placeholder.markdown(html, unsafe_allow_html=True)

    show_steps(0)

    try:
        # Steps 1–3 are instant (Pandas); we call run_full_pipeline which does all
        show_steps(1)
        show_steps(2)
        show_steps(3)

        result = run_full_pipeline(api_key, file_bytes, fhash)
        clusters, metrics, total, port_avg, raw_df, exec_summary = result

        show_steps(4)
        progress_placeholder.empty()

    except Exception as e:
        progress_placeholder.empty()
        st.error(f"Analysis failed: {e}")
        st.stop()

    if clusters is None:
        progress_placeholder.empty()
        st.warning("No issues detected. Check that your file has a review text column.")
        import io as _io
        _df = pd.read_excel(_io.BytesIO(file_bytes))
        _df.columns = [c.strip().lower().replace(" ", "_") for c in _df.columns]
        st.write("**Columns found:**", list(_df.columns))
        st.dataframe(_df.head(5))
        st.stop()

    st.session_state.clusters     = clusters
    st.session_state.metrics      = metrics
    st.session_state.total        = total
    st.session_state.port_avg     = port_avg
    st.session_state.raw_df       = raw_df
    st.session_state.exec_summary = exec_summary
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═══════════════════════════════════════════════════════════════
st.markdown("# Review Intelligence")
st.markdown("Spacez · Guest review analysis powered by Gemini")

if st.session_state.clusters is None:
    st.info("Upload the Spacez review Excel file in the sidebar, then click **▶ Run analysis**.")
    st.stop()

clusters      = st.session_state.clusters
metrics       = st.session_state.metrics
total_rev     = st.session_state.total
port_avg      = st.session_state.port_avg
port_avg5     = round(port_avg * 5, 2)
exec_summary  = st.session_state.exec_summary or ""

# Apply sidebar filters
filtered = clusters[
    clusters["priority"].isin(filter_priority) &
    clusters["owner"].isin(filter_owner)
].copy()

# ── Executive summary banner ──────────────────────────────────
if exec_summary:
    with st.expander("📋 Executive summary", expanded=True):
        st.markdown(exec_summary)

# ── Top metrics row ───────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
with c1: render_metric("Total reviews", total_rev)
with c2: render_metric("Issue clusters", metrics["total_issues"],
                        f"{metrics['high_count']} high · {metrics['med_count']} medium")
with c3: render_metric("Portfolio avg", f"{port_avg5}/5", "Normalised across platforms")
with c4: render_metric("Recurring issues", metrics["recurring_count"], "3+ mentions, same property")
with c5: render_metric("Resolution potential", f"{metrics['issue_resolution_potential']}%",
                        "Of issues are controllable")

st.markdown("---")

tab_ops, tab_biz, tab_care, tab_copilot = st.tabs([
    "🔧 Operations", "📊 Business", "👤 Caretakers", "💬 Copilot"
])


# ═══════════════════════════════════════════════════════════════
# OPERATIONS TAB
# ═══════════════════════════════════════════════════════════════
with tab_ops:
    st.subheader("Operations issue queue")

    col1, col2 = st.columns([2,1])
    with col1:
        pri_counts = filtered["priority"].value_counts().reindex(["High","Medium","Low"], fill_value=0)
        fig = go.Figure(go.Bar(
            x=pri_counts.index, y=pri_counts.values,
            marker_color=["#D94F3D","#E8971A","#3B8A5C"],
            text=pri_counts.values, textposition="outside"
        ))
        fig.update_layout(
            title="Issues by priority", plot_bgcolor="white", paper_bgcolor="white",
            height=260, margin=dict(l=0,r=0,t=40,b=0), showlegend=False,
            font=dict(family="Inter", color="#1A1915")
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ri = metrics.get("rating_impacts", {})
        if ri:
            cats   = list(ri.keys())[:6]
            vals   = [ri[c]*5 for c in cats]
            colors = ["#D94F3D" if v>0.3 else "#E8971A" if v>0.1 else "#3B8A5C" for v in vals]
            fig2   = go.Figure(go.Bar(
                x=vals, y=cats, orientation="h",
                marker_color=colors,
                text=[f"−{round(v,1)}" for v in vals], textposition="outside"
            ))
            fig2.update_layout(
                title="Rating impact by category (/5)", plot_bgcolor="white", paper_bgcolor="white",
                height=260, margin=dict(l=0,r=0,t=40,b=0),
                font=dict(family="Inter", color="#1A1915"),
                xaxis=dict(range=[0, max(vals+[0.1])*1.5])
            )
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown(f"**{len(filtered)} issue clusters** sorted by priority score")
    st.markdown("---")
    for _, row in filtered.sort_values("priority_score", ascending=False).iterrows():
        render_issue_card(row, port_avg5)


# ═══════════════════════════════════════════════════════════════
# BUSINESS TAB
# ═══════════════════════════════════════════════════════════════
with tab_biz:
    st.subheader("Portfolio health overview")

    ph = metrics.get("property_health", {})
    if ph:
        props  = list(ph.keys())
        scores = list(ph.values())
        colors = ["#D94F3D" if s<60 else "#E8971A" if s<80 else "#3B8A5C" for s in scores]
        fig3   = go.Figure(go.Bar(
            x=props, y=scores, marker_color=colors,
            text=scores, textposition="outside"
        ))
        fig3.update_layout(
            title="Property health score (0–100)", plot_bgcolor="white", paper_bgcolor="white",
            height=300, margin=dict(l=0,r=0,t=40,b=0), yaxis=dict(range=[0,115]),
            font=dict(family="Inter", color="#1A1915")
        )
        st.plotly_chart(fig3, use_container_width=True)

    biz_table = clusters.groupby("property").agg(
        total_issues  =("category","count"),
        high_priority =("priority", lambda x: (x=="High").sum()),
        avg_rating    =("avg_rating_normalised", lambda x: round(x.mean()*5,1)),
        rating_impact =("rating_impact", lambda x: round(x.mean()*5,2)),
        recurring     =("is_recurring","sum"),
        health_score  =("property", lambda x: ph.get(x.iloc[0],"-"))
    ).reset_index()
    biz_table.columns = ["Property","Issues","High priority","Avg rating (/5)","Avg impact","Recurring","Health score"]
    biz_table["Risk flag"] = biz_table.apply(
        lambda r: "🔴 Review now" if r["High priority"] >= 2
        else ("🟡 Watch" if r["High priority"] >= 1 or r["Recurring"] >= 2 else "🟢 Stable"), axis=1
    )
    st.dataframe(biz_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        render_metric("Operational risk score", f"{metrics['ops_risk_score']}/100",
                      "High risk" if metrics['ops_risk_score']>60 else
                      ("Moderate" if metrics['ops_risk_score']>30 else "Low risk"))
    with c2:
        render_metric("Person-level patterns", metrics['person_level_count'],
                      "Issue clusters tied to a caretaker across properties")


# ═══════════════════════════════════════════════════════════════
# CARETAKER TAB
# ═══════════════════════════════════════════════════════════════
with tab_care:
    st.subheader("Caretaker performance")

    cs = metrics.get("caretaker_scores", {})
    if not cs:
        st.info("No caretaker data found.")
    else:
        all_ctrl     = sum(v["controllable"]     for v in cs.values())
        all_non_ctrl = sum(v["non_controllable"] for v in cs.values())
        total_ct     = all_ctrl + all_non_ctrl

        m1,m2,m3 = st.columns(3)
        with m1: render_metric("Controllable issues", all_ctrl,
                                f"{round(all_ctrl/max(total_ct,1)*100)}% of complaints")
        with m2: render_metric("Non-controllable", all_non_ctrl, "Infrastructure / location / listing")
        with m3:
            avg_score = round(sum(v["score"] for v in cs.values()) / len(cs), 1)
            render_metric("Avg caretaker score", f"{avg_score}", "Out of 100")

        names = list(cs.keys())
        fig4  = go.Figure()
        fig4.add_trace(go.Bar(name="Controllable",
                              x=names, y=[cs[n]["controllable"] for n in names],
                              marker_color="#D94F3D"))
        fig4.add_trace(go.Bar(name="Non-controllable",
                              x=names, y=[cs[n]["non_controllable"] for n in names],
                              marker_color="#C8C4BC"))
        fig4.update_layout(
            barmode="stack", title="Complaints: controllable vs non-controllable",
            plot_bgcolor="white", paper_bgcolor="white",
            height=300, margin=dict(l=0,r=0,t=40,b=20),
            font=dict(family="Inter", color="#1A1915"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")
        st.caption("Only controllable issues appear below. Infrastructure, location, and listing issues are excluded.")

        for ct_name, ct_data in cs.items():
            ctrl_issues = clusters[
                (clusters["caretaker"] == ct_name) &
                (clusters["caretaker_controllable"] == True)
            ].sort_values("priority_score", ascending=False)

            with st.expander(f"**{ct_name}** · Score {ct_data['score']}/100 · {ct_data['controllable']} controllable issues"):
                c1,c2,c3 = st.columns(3)
                with c1: render_metric("Controllable issues", ct_data["controllable"])
                with c2: render_metric("Non-controllable", ct_data["non_controllable"], "Excluded from score")
                with c3: render_metric("% within control",
                                       f"{round(ct_data['controllable']/max(ct_data['total'],1)*100)}%")

                if not ctrl_issues.empty:
                    for _, ir in ctrl_issues.iterrows():
                        evs  = ir.get("evidence_reviews", [])
                        ev1  = str(evs[0])[:150] if evs else "No quotes available."
                        cls_ = {"High":"badge-high","Medium":"badge-medium","Low":"badge-low"}.get(ir["priority"],"badge-low")
                        st.markdown(f"""
                        <div class="issue-card {ir['priority'].lower()}" style="margin-bottom:.6rem">
                            <div class="issue-title" style="font-size:.9rem">{ir['category']} · {ir['property']}</div>
                            <div class="issue-meta">
                                <span class="badge {cls_}">{ir['priority']}</span>
                                · {ir['frequency']} mentions · {ir['confidence']}% confidence
                            </div>
                            <div class="evidence-quote" style="font-size:.8rem">"{ev1}"</div>
                            <div style="font-size:.8rem;margin-top:.5rem;color:#3A3730">
                                <b>Coaching note:</b> {ir['action_recommendation']}
                            </div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.success("No controllable issues for this caretaker.")


# ═══════════════════════════════════════════════════════════════
# COPILOT TAB  (Req 2 — Gemini only for Q&A)
# ═══════════════════════════════════════════════════════════════
with tab_copilot:
    st.subheader("Operations Copilot")
    st.caption("Ask questions about the analysis. Answers are grounded in the extracted data.")

    summary_str = clusters.to_string(
        columns=["property","category","priority","frequency","confidence",
                 "owner","root_cause","rating_impact","person_level_pattern",
                 "action_recommendation"],
        index=False, max_colwidth=100
    )
    COPILOT_SYS = f"""You are the Spacez Operations Copilot. Answer questions about guest review analysis.
Always cite specific numbers, property names, and caretaker names from the data.
Keep answers concise and actionable. Say so if the data doesn't support a claim.

Analysis data:
{summary_str}

Portfolio average rating (0-1): {round(port_avg, 3)}
Total reviews: {total_rev}
"""

    example_qs = [
        "What are the top recurring issues?",
        "Which property needs most attention?",
        "Any caretaker cross-property patterns?",
    ]
    eq_cols = st.columns(3)
    for i, q in enumerate(example_qs):
        with eq_cols[i]:
            if st.button(q, key=f"eq_{i}", use_container_width=True):
                st.session_state.setdefault("chat_history", [])
                st.session_state["chat_history"].append({"role":"user","content":q})

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if prompt := st.chat_input("Ask the copilot…"):
        st.session_state["chat_history"].append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                copilot_model = get_model(api_key)
                history_text  = "\n".join(
                    f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                    for m in st.session_state["chat_history"][:-1]
                )
                full_prompt = (
                    f"{COPILOT_SYS}\n\nConversation:\n{history_text}"
                    f"\n\nUser: {prompt}\nAssistant:"
                )
                reply = gemini_call(copilot_model, full_prompt, max_tokens=600)
            st.write(reply)
            st.session_state["chat_history"].append({"role":"assistant","content":reply})
