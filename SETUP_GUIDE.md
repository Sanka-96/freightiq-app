# 🚛 FreightIQ Pro — Quick Setup Guide

Single-file Streamlit app with Firebase Auth + Firestore + **Interactive Charts**.

---

## 📦 What's in this ZIP

```
FreightApp.py            ← Main app (single file, all logic inside)
requirements.txt         ← Python dependencies (includes Plotly)
secrets.toml.example     ← Firebase credentials template
firestore.rules          ← Firestore security rules
.gitignore               ← Git ignore (protects credentials)
SETUP_GUIDE.md           ← This file
```

You will add separately:
- `rf_model.pkl` (your trained Random Forest)
- `feature_cols.pkl` (feature column list)

---

## ✨ NEW: History Page with Charts

History page එක දැන් **4 interactive Plotly charts** එක්ක:

| Chart | Type | What it shows |
|-------|------|---------------|
| 📊 **Cost Trend Over Time** | Area Line | Chronological cost trend (oldest → newest) |
| 🚛 **Vehicle Type Distribution** | Donut | Small/Medium/Large vehicle usage % |
| 📍 **Top Routes** | Horizontal Bar | Most predicted origin → destination pairs |
| 📅 **Monthly Predictions** | Combo (Bar + Line) | Predictions count + Total cost per month |

ඔක්කොම charts:
- **Hoverable** — mouse hover කරාම tooltip එක දකින්න
- **Interactive** — zoom, pan, download as PNG
- **Responsive** — mobile/desktop දෙකම look good
- **Professional theme** — purple-pink gradient palette

---

## 🚀 Setup Steps

### 1. Local Testing

```bash
# Extract ZIP and navigate
cd freightiq_single

# Install dependencies
pip install -r requirements.txt

# Add your ML model files (from Colab)
# Place rf_model.pkl and feature_cols.pkl in this folder

# Create secrets file
mkdir .streamlit
cp secrets.toml.example .streamlit/secrets.toml
# → Edit secrets.toml and fill in your real Firebase credentials

# Run the app
streamlit run FreightApp.py
```

Browser opens at `http://localhost:8501` → see login page.

### 2. Streamlit Cloud Deployment

```bash
# Push to GitHub
git init
git add FreightApp.py requirements.txt firestore.rules rf_model.pkl feature_cols.pkl .gitignore
git commit -m "FreightIQ Pro with charts"
git remote add origin https://github.com/Sanka-96/freightiq-pro.git
git push -u origin main
```

Then on https://share.streamlit.io:
1. New app → connect repo `Sanka-96/freightiq-pro`
2. Main file: `FreightApp.py`
3. Deploy
4. After deploy → Settings → **Secrets** tab → paste your secrets.toml contents

Your app will be live at `https://freightiq-pro.streamlit.app`

---

## 🔥 Firebase Setup (Already Done!)

Your Firebase project `freight-iq-pro` is configured. You only need:

### Firestore Rules (one-time)
Firebase Console → Firestore Database → Rules → paste `firestore.rules` content → Publish

### Authorized Domains
Firebase Console → Authentication → Settings → Authorized domains → add:
- `localhost` (already there)
- `freightiq-pro.streamlit.app` (or your actual Streamlit URL)
- `freightiq-pro.netlify.app` (your Netlify landing page)

---

## 📊 Testing the Charts

After login, make **3-5 test predictions** with different routes/vehicles. Then:

1. Click **📊 History** in sidebar
2. See:
   - 4 summary metric cards (Total / Value / Avg Cost / Avg Adjustment)
   - **📈 Analytics Dashboard** with 4 interactive charts
   - **📋 Recent Predictions** list below

The more predictions you make, the more meaningful the charts become.

---

## 🐛 Troubleshooting

### Charts don't show
→ Verify `plotly>=5.18.0` is installed: `pip install --upgrade plotly`

### "ModuleNotFoundError: plotly"
→ Run `pip install -r requirements.txt` again

### Charts show but look broken
→ Hard refresh browser (Ctrl+Shift+R / Cmd+Shift+R)

### Monthly chart says "data will appear after predictions accumulate"
→ Normal — needs at least 2 predictions in different months

---

## 📝 Quick Reference

| Component | Location |
|-----------|----------|
| Charts (new!) | `page_history()` function, lines ~660–790 |
| Authentication | `sign_in_user()`, `sign_up_user()` |
| Save predictions | `save_prediction()` |
| Load history | `get_user_predictions()` |
| ML model loading | `load_model()` |
| Weather API | `get_weather()` |
| Distance API | `get_road_distance()` |

---

*FreightIQ Pro · Thesis ST87628 · TTI Riga · 2026 · @Sanka-96*
