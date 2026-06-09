import streamlit as st
import pandas as pd
import json
import re
import plotly.graph_objects as go
import google.generativeai as genai

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Spacez · Review Intelligence",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background: #F7F5F0; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

.metric-card {
    background: white; border-radius: 12px; padding: 1.2rem 1.4rem;
    border: 1px solid #E8E4DC; margin-bottom: 0.8rem;
}
.metric-card .label {
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.07em;
    text-transform: uppercase; color: #8B8578; margin-bottom: 0.3rem;
}
.metric-card .value { font-family: 'Sora', sans-serif; font-size: 2rem; font-weight: 700; color: #1A1915; line-height: 1; }
.metric-card .delta { font-size: 0.8rem; margin-top: 0.3rem; color: #8B8578; }

.issue-card {
    background: white; border-radius: 12px; padding: 1.4rem;
    border: 1px solid #E8E4DC; margin-bottom: 1rem; border-left: 4px solid #E8E4DC;
}
.issue-card.high   { border-left-color: #D94F3D; }
.issue-card.medium { border-left-color: #E8971A; }
.issue-card.low    { border-left-color: #3B8A5C; }

.issue-title { font-family: 'Sora', sans-serif; font-size: 1rem; font-weight: 700; color: #1A1915; margin-bottom: 0.3rem; }
.issue-meta  { font-size: 0.78rem; color: #8B8578; margin-bottom: 0.8rem; }

.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.7rem; font-weight: 600; margin-right: 6px; letter-spacing: 0.04em; }
.badge-high   { background: #FDECEA; color: #9B2218; }
.badge-medium { background: #FEF3DE; color: #8A5310; }
.badge-low    { background: #E6F4EC; color: #1E6641; }
.badge-ops    { background: #EEF2FF; color: #3730A3; }
.badge-biz    { background: #F0FDF4; color: #14532D; }
.badge-care   { background: #FFF7ED; color: #7C2D12; }
.badge-person { background: #FDF4FF; color: #6B21A8; }

.evidence-box { background: #F7F5F0; border-radius: 8px; padding: 0.8rem 1rem; margin-top: 0.6rem; border-left: 3px solid #C8C4BC; }
.evidence-quote { font-size: 0.82rem; color: #4A4740; font-style: italic; margin-bottom: 0.4rem; padding-left: 0.5rem; border-left: 2px solid #C8C4BC; }
.section-label { font-size: 0.7rem; font-weight: 600; color: #8B8578; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.4rem; }

.explain-row { display: flex; gap: 0.6rem; align-items: flex-start; margin-bottom: 0.4rem; font-size: 0.8rem; }
.explain-step { min-width: 80px; font-weight: 600; color: #6B6560; }
.explain-text { color: #3A3730; }

[data-testid="stSidebar"] { background: #1A1915; }
[data-testid="stSidebar"] * { color: #E8E4DC !important; }
[data-testid="stSidebar"] .stTextInput input { background: #2C2A25; border-color: #3A3730; color: #E8E4DC; }
[data-testid="stSidebar"] hr { border-color: #3A3730; }

h1, h2, h3 { font-family: 'Sora', sans-serif; font-weight: 700; color: #1A1915; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────
SEVERITY_SCORE = {"High": 3, "Medium": 2, "Low": 1}
OWNER_URGENCY  = {"Operations": 1.2, "Caretaker": 1.1, "Business": 1.0, "Vendor": 0.9, "Unknown": 0.8}

# ── Gemini client helper ──────────────────────────────────────
def get_model(api_key: str, model_name: str = "gemini-1.5-flash"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)

def gemini_call(model, prompt: str, max_tokens: int = 1000) -> str:
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(max_output_tokens=max_tokens, temperature=0.2)
    )
    return response.text.strip()

# ── Step 1: Load & normalise ──────────────────────────────────
def load_reviews(uploaded_file) -> pd.DataFrame:
    df = pd.read_excel(uploaded_file)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

def normalise_ratings(df: pd.DataFrame) -> pd.DataFrame:
    def norm(row):
        platform = str(row.get("platform", "")).lower()
        rating   = row.get("rating", None)
        if pd.isna(rating):
            return None
        return (rating / 10.0) if "booking" in platform else (rating / 5.0)
    df["normalised_rating"] = df.apply(norm, axis=1)
    return df

def detect_columns(df: pd.DataFrame):
    cols = df.columns.tolist()
    review_col    = next((c for c in cols if any(k in c for k in ["review","comment","text","feedback"])), cols[0])
    property_col  = next((c for c in cols if "property" in c), None)
    caretaker_col = next((c for c in cols if "caretaker" in c or "host" in c), None)
    return review_col, property_col, caretaker_col

def filter_noise(df: pd.DataFrame, review_col: str) -> pd.DataFrame:
    NOISE = re.compile(r"^(amazing|perfect|great|loved it|excellent|fantastic|wonderful|5 stars)[!.]?$", re.I)
    df["is_noise"] = (
        df[review_col].fillna("").str.strip().str.len() < 15
    ) | df[review_col].fillna("").str.strip().str.match(NOISE)
    return df

# ── Step 2: AI issue extraction ───────────────────────────────
EXTRACT_PROMPT = """You are an operations intelligence system for Spacez, a luxury villa company.

Analyse this guest review and extract structured operational intelligence.

Review: {review_text}
Platform: {platform}
Normalised rating (0-1 scale): {rating}
Property: {property}
Caretaker: {caretaker}

Reply ONLY with valid JSON. No markdown fences, no preamble, no explanation.

{{
  "has_actionable_issue": true,
  "issues": [
    {{
      "category": "Housekeeping|Maintenance|Check-in|Staff Behaviour|Amenities|Cleanliness|Safety|Communication|Other",
      "description": "concise description of the issue",
      "severity": "Low|Medium|High",
      "caretaker_controllable": true,
      "non_controllable_reason": "if not controllable, reason here",
      "likely_cause": "brief hypothesis",
      "owner": "Operations|Caretaker|Business|Vendor|Unknown"
    }}
  ],
  "positive_highlights": ["list of genuine positives"],
  "noise_review": false
}}"""

def extract_issues(row, review_col: str, model) -> dict:
    prompt = EXTRACT_PROMPT.format(
        review_text=str(row[review_col])[:1200],
        platform=row.get("platform", "Unknown"),
        rating=round(float(row.get("normalised_rating", 0.5)), 2),
        property=row.get("property", row.get("property_name", "Unknown")),
        caretaker=row.get("caretaker", row.get("caretaker_name", "Unknown")),
    )
    try:
        raw = gemini_call(model, prompt, max_tokens=1000)
        raw = re.sub(r"```json|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        return {"has_actionable_issue": False, "issues": [], "error": str(e)}

# ── Step 3: Pattern detection with evidence ───────────────────
def detect_patterns(extracted: list) -> pd.DataFrame:
    records = []
    for item in extracted:
        if not item.get("has_actionable_issue"):
            continue
        for issue in item.get("issues", []):
            records.append({
                "property":               item.get("property", "Unknown"),
                "caretaker":              item.get("caretaker", "Unknown"),
                "category":               issue.get("category", "Other"),
                "description":            issue.get("description", ""),
                "severity":               issue.get("severity", "Medium"),
                "caretaker_controllable": issue.get("caretaker_controllable", True),
                "non_controllable_reason":issue.get("non_controllable_reason", ""),
                "likely_cause":           issue.get("likely_cause", ""),
                "owner":                  issue.get("owner", "Unknown"),
                "review_text":            item.get("review_text", ""),
                "normalised_rating":      item.get("normalised_rating", 0.5),
            })

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # Frequency per property+category
    freq = df.groupby(["property","category"]).size().reset_index(name="frequency")
    freq["is_recurring"] = freq["frequency"] >= 3
    df = df.merge(freq, on=["property","category"])

    # Avg rating per cluster
    rating_avg = df.groupby(["property","category"])["normalised_rating"].mean().reset_index(name="avg_rating_normalised")
    df = df.merge(rating_avg, on=["property","category"])

    # Person-level pattern
    cross = (
        df[df["caretaker_controllable"]==True]
        .groupby(["caretaker","category"])["property"].nunique()
        .reset_index(name="property_count")
    )
    cross["person_level_pattern"] = cross["property_count"] >= 2
    df = df.merge(cross[["caretaker","category","person_level_pattern"]], on=["caretaker","category"], how="left")
    df["person_level_pattern"] = df["person_level_pattern"].fillna(False)

    # Evidence collection (up to 3 quotes per cluster)
    def collect_evidence(group):
        texts = group["review_text"].dropna().tolist()
        return sorted(texts, key=len)[:3]

    evidence_map = df.groupby(["property","category"]).apply(collect_evidence).reset_index(name="evidence_reviews")
    df = df.merge(evidence_map, on=["property","category"])

    return df

# ── Step 4: Aggregated root cause ─────────────────────────────
ROOT_CAUSE_PROMPT = """You are an operations analyst for Spacez luxury villas.

Generate a specific, evidence-based root cause for this issue cluster. Cite the actual numbers.

Issue: {category}
Property: {property}
Frequency: {frequency} reviews
Is recurring (3+ occurrences): {is_recurring}
Person-level pattern (same caretaker, multiple properties): {person_level}
Caretaker: {caretaker}
Sample descriptions: {descriptions}
Average normalised rating: {avg_rating}

Write ONE or TWO sentences. Cite specific numbers.
Example: "7 delayed check-in complaints across 4 properties managed by [name] suggest a scheduling pattern, not a property issue."
Reply in plain text only. No JSON."""

def generate_root_cause(cluster: dict, model) -> str:
    prompt = ROOT_CAUSE_PROMPT.format(
        category=cluster["category"],
        property=cluster["property"],
        frequency=cluster["frequency"],
        is_recurring=cluster["is_recurring"],
        person_level=cluster["person_level_pattern"],
        caretaker=cluster["caretaker"],
        descriptions="; ".join(cluster.get("descriptions", [])[:3]),
        avg_rating=round(cluster["avg_rating_normalised"], 2),
    )
    try:
        return gemini_call(model, prompt, max_tokens=200)
    except:
        return cluster.get("likely_cause", "Pattern detected across multiple reviews.")

# ── Step 5: Priority + confidence ────────────────────────────
def compute_priority_score(row, portfolio_avg):
    freq_score     = min(row["frequency"] / 3, 3.0)
    severity_score = SEVERITY_SCORE.get(row["severity"], 1)
    owner_mult     = OWNER_URGENCY.get(row["owner"], 1.0)
    recurring_mult = 1.5 if row["is_recurring"] else 1.0
    person_mult    = 1.3 if row.get("person_level_pattern", False) else 1.0
    rating_impact  = max(0, portfolio_avg - row["avg_rating_normalised"])
    rating_mult    = 1.0 + (rating_impact * 2)
    return round(freq_score * severity_score * owner_mult * recurring_mult * person_mult * rating_mult, 2)

def priority_label(score):
    if score >= 10: return "High"
    elif score >= 4: return "Medium"
    return "Low"

def compute_confidence(row, total_reviews):
    freq_signal      = min(row["frequency"] / max(total_reviews * 0.08, 1), 1.0) * 55
    recurrence_bonus = 20 if row["is_recurring"] else 0
    person_bonus     = 15 if row.get("person_level_pattern", False) else 0
    severity_bonus   = {"High": 8, "Medium": 4, "Low": 0}.get(row["severity"], 0)
    evidence_bonus   = min(len(row.get("evidence_reviews", [])) * 3, 9)
    return min(int(freq_signal + recurrence_bonus + person_bonus + severity_bonus + evidence_bonus), 99)

# ── Step 6: Action recommendation ────────────────────────────
ACTION_PROMPT = """You are an operations advisor for Spacez luxury villas.

Issue: {category} at {property}
Frequency: {frequency} reviews
Root cause: {root_cause}
Owner: {owner}
Priority: {priority}
Caretaker controllable: {controllable}
Person-level pattern: {person_level}

Write ONE specific action recommendation (2-3 sentences).
Be concrete: who does what, by when, what the measurable outcome is.
No vague phrases like "improve" or "address the issue".
Reply in plain text only."""

def generate_action(cluster: dict, model) -> str:
    try:
        return gemini_call(model, ACTION_PROMPT.format(**cluster), max_tokens=250)
    except Exception as e:
        return f"Action generation failed: {e}"

# ── Step 7: Business metrics ──────────────────────────────────
def compute_business_metrics(clusters: pd.DataFrame, total_reviews: int, portfolio_avg: float) -> dict:
    high_issues   = len(clusters[clusters["priority"] == "High"])
    med_issues    = len(clusters[clusters["priority"] == "Medium"])
    total_issues  = len(clusters)
    recurring_cnt = int(clusters["is_recurring"].sum())
    person_cnt    = int(clusters["person_level_pattern"].sum())

    controllable_pct = clusters["caretaker_controllable"].mean() if total_issues else 0

    property_scores = {}
    for prop, grp in clusters.groupby("property"):
        penalty = grp.apply(
            lambda r: {"High":15,"Medium":7,"Low":2}.get(r["priority"],0) * (1.5 if r["is_recurring"] else 1.0),
            axis=1
        ).sum()
        property_scores[prop] = max(0, round(100 - penalty, 1))

    ops_risk = min(100, round((high_issues*15 + med_issues*5 + recurring_cnt*8) / max(total_issues,1) * 10, 1))

    caretaker_scores = {}
    for ct, grp in clusters.groupby("caretaker"):
        ctrl  = int(grp["caretaker_controllable"].sum())
        total = len(grp)
        penalty = grp[grp["caretaker_controllable"]==True].apply(
            lambda r: {"High":10,"Medium":5,"Low":1}.get(r["priority"],0), axis=1
        ).sum()
        pct   = ctrl / total if total else 0
        score = max(0, round(100 - penalty * pct, 1))
        caretaker_scores[ct] = {"score": score, "controllable": ctrl, "non_controllable": total-ctrl, "total": total}

    rating_impacts = {}
    for cat, grp in clusters.groupby("category"):
        rating_impacts[cat] = round(portfolio_avg - grp["avg_rating_normalised"].mean(), 3)

    return {
        "total_issues": total_issues,
        "high_count": high_issues,
        "med_count": med_issues,
        "recurring_count": recurring_cnt,
        "person_level_count": person_cnt,
        "issue_resolution_potential": round(controllable_pct * 100, 1),
        "property_health": property_scores,
        "ops_risk_score": ops_risk,
        "caretaker_scores": caretaker_scores,
        "rating_impacts": rating_impacts,
    }

# ── Main pipeline ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_full_pipeline(_api_key: str, file_bytes: bytes, filename: str):
    import io
    model = get_model(_api_key)

    df = load_reviews(io.BytesIO(file_bytes))
    df = normalise_ratings(df)
    review_col, property_col, caretaker_col = detect_columns(df)
    df = filter_noise(df, review_col)

    actionable    = df[~df["is_noise"]].reset_index(drop=True)
    total_reviews = len(df)
    portfolio_avg = df["normalised_rating"].dropna().mean()

    extracted = []
    progress  = st.progress(0, text="Extracting issues from reviews…")
    for i, row in actionable.iterrows():
        result = extract_issues(row, review_col, model)
        result["review_text"]       = str(row[review_col])[:300]
        result["property"]          = str(row.get(property_col, "Unknown")) if property_col else "Unknown"
        result["caretaker"]         = str(row.get(caretaker_col, "Unknown")) if caretaker_col else "Unknown"
        result["normalised_rating"] = float(row.get("normalised_rating", 0.5))
        extracted.append(result)
        progress.progress(int((i+1) / len(actionable) * 60), text=f"Extracting issues… {i+1}/{len(actionable)}")

    issues_df = detect_patterns(extracted)
    if issues_df.empty:
        progress.empty()
        return None, None, total_reviews, portfolio_avg, df

    clusters = (
        issues_df.sort_values("frequency", ascending=False)
        .drop_duplicates(subset=["property","category"])
        .reset_index(drop=True)
    )

    desc_map  = issues_df.groupby(["property","category"])["description"].apply(list).reset_index(name="descriptions")
    clusters  = clusters.merge(desc_map, on=["property","category"])
    clusters["rating_impact"]   = clusters["avg_rating_normalised"].apply(lambda r: round(portfolio_avg - r, 3))
    clusters["priority_score"]  = clusters.apply(lambda r: compute_priority_score(r, portfolio_avg), axis=1)
    clusters["priority"]        = clusters["priority_score"].apply(priority_label)
    clusters["confidence"]      = clusters.apply(lambda r: compute_confidence(r, total_reviews), axis=1)

    root_causes, actions = [], []
    total_clusters = len(clusters)
    for idx, row in clusters.iterrows():
        progress.progress(60 + int((idx+1) / total_clusters * 40), text=f"Generating root causes & actions… {idx+1}/{total_clusters}")
        rc = generate_root_cause(row.to_dict(), model)
        ac = generate_action({
            "category": row["category"], "property": row["property"],
            "frequency": row["frequency"], "root_cause": rc,
            "owner": row["owner"], "priority": row["priority"],
            "controllable": row["caretaker_controllable"],
            "person_level": row.get("person_level_pattern", False),
        }, model)
        root_causes.append(rc)
        actions.append(ac)

    clusters["root_cause"]            = root_causes
    clusters["action_recommendation"] = actions
    progress.empty()

    metrics = compute_business_metrics(clusters, total_reviews, portfolio_avg)
    return clusters, metrics, total_reviews, portfolio_avg, df

# ── UI helpers ────────────────────────────────────────────────
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
    p    = row["priority"]
    cls  = p.lower()
    evs  = row.get("evidence_reviews", [])

    ev_html = ""
    for ev in evs[:3]:
        if ev and len(str(ev).strip()) > 10:
            short = str(ev).strip()[:180] + ("…" if len(str(ev).strip()) > 180 else "")
            ev_html += f'<div class="evidence-quote">"{short}"</div>'

    person_badge = '<span class="badge badge-person">⚠ person-level pattern</span>' if row.get("person_level_pattern") else ""
    rating5      = round(row["avg_rating_normalised"] * 5, 1)
    impact5      = round(row["rating_impact"] * 5, 1)
    impact_str   = f"−{impact5}" if impact5 > 0 else f"+{abs(impact5)}"

    explain_html = f"""
    <div class="explain-row"><span class="explain-step">Detected:</span><span class="explain-text">{row['frequency']} reviews mention this in {row['property']}</span></div>
    <div class="explain-row"><span class="explain-step">Priority:</span><span class="explain-text">{p} — freq {row['frequency']} × severity {row['severity']} × rating impact {impact_str}/5</span></div>
    <div class="explain-row"><span class="explain-step">Owner:</span><span class="explain-text">{row['owner']} — {'caretaker-controllable' if row['caretaker_controllable'] else 'outside caretaker control'}</span></div>
    <div class="explain-row"><span class="explain-step">Action:</span><span class="explain-text">Addresses root cause pattern across {row['frequency']} reviews</span></div>
    """

    st.markdown(f"""
    <div class="issue-card {cls}">
        <div class="issue-title">{row['category']} · {row['property']}</div>
        <div class="issue-meta">
            {priority_badge(p)} {owner_badge(row['owner'])} {person_badge}
            &nbsp;·&nbsp; {row['frequency']} reviews &nbsp;·&nbsp; Confidence {row['confidence']}%
            &nbsp;·&nbsp; Avg rating {rating5}/5 (portfolio {port_avg5}/5)
            &nbsp;·&nbsp; Rating impact <b>{impact_str}/5</b>
        </div>

        <div class="section-label">Root cause</div>
        <div style="font-size:0.85rem;color:#3A3730;margin-bottom:0.8rem;">{row['root_cause']}</div>

        <div class="section-label">Evidence reviews ({min(len(evs),3)} shown)</div>
        <div class="evidence-box">{ev_html or '<span style="font-size:0.8rem;color:#8B8578;">No direct quotes stored.</span>'}</div>

        <div class="section-label" style="margin-top:0.8rem;">Recommended action</div>
        <div style="font-size:0.85rem;color:#1A1915;font-weight:500;margin-bottom:0.8rem;">{row['action_recommendation']}</div>

        <div class="section-label">Explainability</div>
        {explain_html}
    </div>""", unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏡 Spacez")
    st.markdown("**Review Intelligence System**")
    st.markdown("---")
    api_key  = st.text_input("Google Gemini API key", type="password", placeholder="AIza…")
    st.caption("Get a free key at [aistudio.google.com](https://aistudio.google.com/app/apikey)")
    uploaded = st.file_uploader("Upload review Excel file", type=["xlsx","xls"])
    run_btn  = st.button("Run analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown("**Filters**")
    filter_priority = st.multiselect("Priority", ["High","Medium","Low"], default=["High","Medium","Low"])
    filter_owner    = st.multiselect("Owner", ["Operations","Caretaker","Business","Vendor","Unknown"],
                                     default=["Operations","Caretaker","Business","Vendor","Unknown"])
    st.markdown("---")
    st.caption("Powered by Google Gemini 1.5 Flash")

# ── STATE ──────────────────────────────────────────────────────
for key in ["clusters","metrics","total","port_avg","raw_df"]:
    if key not in st.session_state:
        st.session_state[key] = None

if run_btn:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()
    if not uploaded:
        st.error("Please upload a review Excel file.")
        st.stop()
    clusters, metrics, total, port_avg, raw_df = run_full_pipeline(
        api_key, uploaded.read(), uploaded.name
    )
    st.session_state.clusters = clusters
    st.session_state.metrics  = metrics
    st.session_state.total    = total
    st.session_state.port_avg = port_avg
    st.session_state.raw_df   = raw_df

# ── MAIN CONTENT ───────────────────────────────────────────────
st.markdown("# Review Intelligence")
st.markdown("Spacez · Guest review analysis powered by Gemini")

if st.session_state.clusters is None:
    st.info("Enter your **Gemini API key**, upload the review Excel file, then click **Run analysis**.")
    st.markdown("""
    **How to get a free Gemini API key:**
    1. Go to [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
    2. Sign in with Google
    3. Click **Create API key** — it's free
    """)
    st.stop()

clusters  = st.session_state.clusters
metrics   = st.session_state.metrics
total_rev = st.session_state.total
port_avg  = st.session_state.port_avg
port_avg5 = round(port_avg * 5, 2)

filtered = clusters[
    clusters["priority"].isin(filter_priority) &
    clusters["owner"].isin(filter_owner)
].copy()

# ── TOP METRICS ────────────────────────────────────────────────
c1,c2,c3,c4,c5 = st.columns(5)
with c1: render_metric("Total reviews", total_rev)
with c2: render_metric("Issue clusters", metrics["total_issues"], f"{metrics['high_count']} high · {metrics['med_count']} medium")
with c3: render_metric("Portfolio avg", f"{port_avg5}/5", "Normalised across platforms")
with c4: render_metric("Recurring issues", metrics["recurring_count"], "3+ mentions, same property")
with c5: render_metric("Resolution potential", f"{metrics['issue_resolution_potential']}%", "Of issues are controllable")

st.markdown("---")

tab_ops, tab_biz, tab_care, tab_copilot = st.tabs([
    "🔧 Operations", "📊 Business", "👤 Caretakers", "💬 Copilot"
])

# ══════════════════════════════════════════════
# OPERATIONS TAB
# ══════════════════════════════════════════════
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
            colors = ["#D94F3D" if v > 0.3 else "#E8971A" if v > 0.1 else "#3B8A5C" for v in vals]
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

# ══════════════════════════════════════════════
# BUSINESS TAB
# ══════════════════════════════════════════════
with tab_biz:
    st.subheader("Portfolio health overview")

    ph = metrics.get("property_health", {})
    if ph:
        props  = list(ph.keys())
        scores = list(ph.values())
        colors = ["#D94F3D" if s < 60 else "#E8971A" if s < 80 else "#3B8A5C" for s in scores]
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
        avg_rating    =("avg_rating_normalised", lambda x: round(x.mean()*5, 1)),
        rating_impact =("rating_impact", lambda x: round(x.mean()*5, 2)),
        recurring     =("is_recurring","sum"),
        health_score  =("property", lambda x: ph.get(x.iloc[0], "-"))
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
                      "High risk" if metrics['ops_risk_score'] > 60 else ("Moderate" if metrics['ops_risk_score'] > 30 else "Low risk"))
    with c2:
        render_metric("Person-level patterns", metrics['person_level_count'],
                      "Issue clusters tied to a caretaker across properties")

# ══════════════════════════════════════════════
# CARETAKER TAB
# ══════════════════════════════════════════════
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
        with m1: render_metric("Controllable issues", all_ctrl, f"{round(all_ctrl/max(total_ct,1)*100)}% of complaints")
        with m2: render_metric("Non-controllable",    all_non_ctrl, "Infrastructure / location / listing")
        with m3:
            avg_score = round(sum(v["score"] for v in cs.values()) / len(cs), 1)
            render_metric("Avg caretaker score", f"{avg_score}", "Out of 100")

        names = list(cs.keys())
        fig4  = go.Figure()
        fig4.add_trace(go.Bar(name="Controllable",     x=names, y=[cs[n]["controllable"]     for n in names], marker_color="#D94F3D"))
        fig4.add_trace(go.Bar(name="Non-controllable", x=names, y=[cs[n]["non_controllable"] for n in names], marker_color="#C8C4BC"))
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
                (clusters["caretaker"]==ct_name) &
                (clusters["caretaker_controllable"]==True)
            ].sort_values("priority_score", ascending=False)

            with st.expander(f"**{ct_name}** · Score {ct_data['score']}/100 · {ct_data['controllable']} controllable issues"):
                c1,c2,c3 = st.columns(3)
                with c1: render_metric("Controllable issues", ct_data["controllable"])
                with c2: render_metric("Non-controllable",    ct_data["non_controllable"], "Excluded from score")
                with c3: render_metric("% within control",    f"{round(ct_data['controllable']/max(ct_data['total'],1)*100)}%")

                if not ctrl_issues.empty:
                    for _, ir in ctrl_issues.iterrows():
                        evs = ir.get("evidence_reviews", [])
                        ev1 = str(evs[0])[:150] if evs else "No quotes available."
                        cls_ = {"High":"badge-high","Medium":"badge-medium","Low":"badge-low"}.get(ir["priority"],"badge-low")
                        st.markdown(f"""
                        <div class="issue-card {ir['priority'].lower()}" style="margin-bottom:0.6rem;">
                            <div class="issue-title" style="font-size:0.9rem;">{ir['category']} · {ir['property']}</div>
                            <div class="issue-meta"><span class="badge {cls_}">{ir['priority']}</span> · {ir['frequency']} mentions · {ir['confidence']}% confidence</div>
                            <div class="evidence-quote" style="font-size:0.8rem;">"{ev1}"</div>
                            <div style="font-size:0.8rem;margin-top:0.5rem;color:#3A3730;"><b>Coaching note:</b> {ir['action_recommendation']}</div>
                        </div>""", unsafe_allow_html=True)
                else:
                    st.success("No controllable issues for this caretaker.")

# ══════════════════════════════════════════════
# COPILOT TAB
# ══════════════════════════════════════════════
with tab_copilot:
    st.subheader("Operations Copilot")
    st.caption("Ask questions about the analysis. All answers cite data from the extracted issue clusters.")

    summary_str = clusters.to_string(
        columns=["property","category","priority","frequency","confidence",
                 "owner","root_cause","rating_impact","person_level_pattern","action_recommendation"],
        index=False, max_colwidth=100
    )
    COPILOT_SYS = f"""You are the Spacez Operations Copilot. Answer questions about guest review analysis for a portfolio of luxury villas.
Always cite specific numbers, property names, and caretaker names from the data.
Keep answers concise and actionable. If data doesn't support a claim, say so.

Analysis data:
{summary_str}

Portfolio average rating (0-1 normalised): {round(port_avg, 3)}
Total reviews analysed: {total_rev}
"""

    example_qs = [
        "What are the top recurring issues?",
        "Which property needs most attention?",
        "Any caretaker cross-property patterns?",
    ]
    cols = st.columns(3)
    for i, q in enumerate(example_qs):
        with cols[i]:
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
                # Build conversation context for Gemini (stateless per call)
                history_text = "\n".join(
                    f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}"
                    for m in st.session_state["chat_history"][:-1]
                )
                full_prompt = f"{COPILOT_SYS}\n\nConversation so far:\n{history_text}\n\nUser: {prompt}\nAssistant:"
                reply = gemini_call(copilot_model, full_prompt, max_tokens=600)
            st.write(reply)
            st.session_state["chat_history"].append({"role":"assistant","content":reply})
