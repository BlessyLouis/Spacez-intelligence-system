# Spacez — AI Review Intelligence System
### Powered by Google Gemini 1.5 Flash (free tier)

---

## Get your free Gemini API key (2 minutes)

1. Go to **https://aistudio.google.com/app/apikey**
2. Sign in with any Google account
3. Click **Create API key**
4. Copy the key — it starts with `AIza...`

The free tier allows ~1,500 requests/day — more than enough for this prototype.

---

## Deploy to Streamlit Cloud (get a shareable link)

### Option A — GitHub + Streamlit Cloud (recommended)
```bash
# In this folder:
git init
git add .
git commit -m "Spacez review intelligence"
gh repo create spacez-review-intelligence --public --push

# Then:
# 1. Go to https://share.streamlit.io
# 2. New app → connect your repo → Main file: app.py → Deploy
# You'll get a public URL in ~2 minutes
```

### Option B — Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Using the app

1. Paste your **Gemini API key** in the sidebar
2. Upload the **Spacez review Excel file**
3. Click **Run analysis** — takes 1–3 minutes depending on dataset size
4. Explore all four tabs

---

## What each tab shows

| Tab | Audience | Key content |
|-----|----------|-------------|
| 🔧 Operations | Ops team | Issue queue, root causes, evidence quotes, rating impact, explainability |
| 📊 Business | Leadership | Property health scores, risk table, ops risk score |
| 👤 Caretakers | HR / caretakers | Controllable vs non-controllable split, coaching notes |
| 💬 Copilot | Anyone | Natural language Q&A over the full analysis |

---

## All 7 improvements

1. **Evidence reviews** — every issue shows up to 3 supporting quotes
2. **Rating impact** — avg rating per issue vs portfolio avg, feeds priority scoring
3. **Aggregated root cause** — cites specific counts and cross-property patterns
4. **Controllable vs non-controllable** — caretaker tab separates what they can/can't influence
5. **Three stakeholder dashboards** — different views for Ops, Business, Caretakers
6. **Explainability layer** — every card shows why detected, why priority, why owner, why action
7. **Product metrics** — Property Health Score, Ops Risk Score, Caretaker Effectiveness, Resolution Potential
