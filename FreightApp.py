"""
═══════════════════════════════════════════════════════════════════
  FreightIQ Pro — Single-File Edition
  Real-Time Cost Prediction for Sri Lankan Freight Transport

  Author: Jayathunga Kamkanamge Ridma Sanka (ST87628)
  TTI Riga · 2026

  Features:
    - Professional UI with premium design
    - Firebase Authentication (signup/signin)
    - Firestore Database (per-user prediction history)
    - Random Forest model (R²=0.9961, MAPE=3.54%)
    - Open-Meteo Weather API + OSRM Road Distance
═══════════════════════════════════════════════════════════════════
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import pickle
import base64
import math
from pathlib import Path
from collections import Counter

import plotly.express as px
import plotly.graph_objects as go

import firebase_admin
from firebase_admin import credentials, auth, firestore


# ═══════════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ═══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="FreightIQ Pro | Cost Predictor",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ═══════════════════════════════════════════════════════════════════
#  CONSTANTS
# ═══════════════════════════════════════════════════════════════════
SL_CITIES = {
    "Colombo":      {"lat": 6.9271,  "lon": 79.8612},
    "Kandy":        {"lat": 7.2906,  "lon": 80.6337},
    "Galle":        {"lat": 6.0535,  "lon": 80.2210},
    "Trincomalee":  {"lat": 8.5874,  "lon": 81.2152},
    "Jaffna":       {"lat": 9.6615,  "lon": 80.0255},
    "Matara":       {"lat": 5.9549,  "lon": 80.5550},
    "Anuradhapura": {"lat": 8.3114,  "lon": 80.4037},
    "Kurunegala":   {"lat": 7.4867,  "lon": 80.3647},
    "Ratnapura":    {"lat": 6.6828,  "lon": 80.3992},
    "Badulla":      {"lat": 6.9934,  "lon": 81.0550},
    "Hambantota":   {"lat": 6.1241,  "lon": 81.1185},
    "Negombo":      {"lat": 7.2083,  "lon": 79.8358},
    "Gampaha":      {"lat": 7.0917,  "lon": 80.0000},
    "Kalutara":     {"lat": 6.5854,  "lon": 79.9607},
    "Vavuniya":     {"lat": 8.7514,  "lon": 80.4972},
    "Batticaloa":   {"lat": 7.7102,  "lon": 81.6924},
    "Puttalam":     {"lat": 8.0362,  "lon": 79.8283},
    "Matale":       {"lat": 7.4675,  "lon": 80.6234},
    "Nuwara Eliya": {"lat": 6.9497,  "lon": 80.7891},
    "Polonnaruwa":  {"lat": 7.9403,  "lon": 81.0188},
    "Monaragala":   {"lat": 6.8728,  "lon": 81.3507},
    "Kegalle":      {"lat": 7.2513,  "lon": 80.3464},
    "Ampara":       {"lat": 7.2980,  "lon": 81.6747},
    "Dehiwala":     {"lat": 6.8500,  "lon": 79.8650},
    "Moratuwa":     {"lat": 6.7730,  "lon": 79.8816},
    "Kelaniya":     {"lat": 6.9553,  "lon": 79.9219},
    "Kaduwela":     {"lat": 6.9333,  "lon": 79.9833},
    "Maharagama":   {"lat": 6.8480,  "lon": 79.9265},
}
KM_RATES    = {0: 86.0,  1: 110.0, 2: 227.0}
FUEL_CONS   = {0: 9.0,   1: 15.0,  2: 23.0}
VH_NAMES    = {0: "Small (7T)", 1: "Medium (14T)", 2: "Large (24T)"}
VH_AVG_COST = {0: 16912, 1: 35319, 2: 71413}
AVG_KM      = 276.93
URBAN_CITIES = {"Colombo","Negombo","Gampaha","Dehiwala",
                "Moratuwa","Kelaniya","Kaduwela","Maharagama"}
FUEL_PRICES = {
    "2022-01":176,"2022-02":176,"2022-03":250,"2022-04":295,"2022-05":350,
    "2022-06":400,"2022-07":420,"2022-08":420,"2022-09":385,"2022-10":370,
    "2022-11":360,"2022-12":355,"2023-01":350,"2023-02":340,"2023-03":330,
    "2023-04":325,"2023-05":320,"2023-06":310,"2023-07":305,"2023-08":310,
    "2023-09":330,"2023-10":325,"2023-11":320,"2023-12":315,"2024-01":315,
    "2024-02":310,"2024-03":307,"2024-04":315,"2024-05":320,"2024-06":318,
    "2024-07":315,"2024-08":312,"2024-09":308,"2024-10":305,"2024-11":302,
    "2024-12":300,"2025-01":305,"2025-02":310,"2025-03":308,"2025-04":305,
    "2025-05":302,"2025-06":300,"2025-07":298,"2025-08":300,"2025-09":305,
    "2025-10":308,"2025-11":310,"2025-12":312,"2026-01":315,"2026-02":318,
    "2026-03":320,"2026-04":322,"2026-05":325,"2026-06":322,"2026-07":320,
    "2026-08":318,"2026-09":315,"2026-10":313,"2026-11":310,"2026-12":308,
}


# ═══════════════════════════════════════════════════════════════════
#  CSS — Professional Design System
# ═══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
#MainMenu,footer,header{visibility:hidden;}
.stDeployButton{display:none;}
.block-container{padding-top:2rem;max-width:1280px;}
html,body,[class*="css"]{font-family:-apple-system,BlinkMacSystemFont,'Inter','Segoe UI',Roboto,sans-serif;}

.app-header{background:linear-gradient(135deg,#4f46e5 0%,#4338ca 100%);padding:28px 36px;border-radius:16px;margin-bottom:28px;color:white;box-shadow:0 10px 25px -5px rgba(0,0,0,0.1);}
.app-header h1{margin:0;font-size:28px;font-weight:700;color:white;}
.app-header p{margin:6px 0 0 0;font-size:14px;color:rgba(255,255,255,0.85);}

.card{background:white;border:1px solid #e5e7eb;border-radius:14px;padding:24px;margin-bottom:20px;box-shadow:0 1px 2px 0 rgba(0,0,0,0.05);}
.card-title{font-size:16px;font-weight:700;color:#111827;margin-bottom:16px;}

.metric-card{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:16px;height:100%;}
.metric-card.primary{background:linear-gradient(135deg,#eef2ff 0%,white 100%);border-color:#e0e7ff;}
.metric-label{font-size:11px;text-transform:uppercase;letter-spacing:0.05em;font-weight:600;color:#6b7280;margin-bottom:6px;}
.metric-value{font-size:24px;font-weight:700;color:#111827;margin-bottom:4px;}
.metric-value.primary{color:#4f46e5;}
.metric-delta{font-size:12px;color:#6b7280;}

.badge-dry,.badge-rainy,.badge-heavy{padding:5px 12px;border-radius:999px;font-size:11px;font-weight:700;}
.badge-dry{background:#dcfce7;color:#166534;}
.badge-rainy{background:#dbeafe;color:#1e40af;}
.badge-heavy{background:#fef9c3;color:#854d0e;}

.section-title{font-size:14px;font-weight:700;color:#374151;margin:24px 0 12px 0;text-transform:uppercase;letter-spacing:0.05em;}

.factor-row{display:flex;justify-content:space-between;align-items:center;padding:10px 0;border-bottom:1px solid #f3f4f6;font-size:13px;}
.factor-row:last-child{border-bottom:none;}
.factor-label{color:#6b7280;font-weight:500;}
.factor-value{color:#111827;font-weight:600;text-align:right;}

.result-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;}
.result-route{font-size:20px;font-weight:700;color:#111827;}
.result-meta{font-size:12px;color:#6b7280;margin-bottom:20px;}

.model-tag{background:#eef2ff;color:#4338ca;padding:12px 16px;border-radius:10px;font-size:12px;font-weight:600;margin-top:20px;text-align:center;}

.stButton > button{border-radius:10px;padding:12px 24px;font-weight:600;font-size:14px;border:none;}
.stButton > button[kind="primary"]{background:linear-gradient(135deg,#4f46e5,#4338ca);color:white;box-shadow:0 4px 12px rgba(99,102,241,0.3);}

.auth-card{background:white;border-radius:20px;padding:40px;box-shadow:0 10px 25px -5px rgba(0,0,0,0.1);border:1px solid #e5e7eb;max-width:440px;margin:40px auto;}
.auth-logo{text-align:center;margin-bottom:28px;}
.auth-logo .icon{font-size:48px;margin-bottom:12px;display:block;}
.auth-logo h1{font-size:28px;font-weight:800;margin:0;background:linear-gradient(135deg,#4f46e5,#4338ca);-webkit-background-clip:text;-webkit-text-fill-color:transparent;}
.auth-logo p{font-size:13px;color:#6b7280;margin:6px 0 0 0;}
.auth-title{font-size:20px;font-weight:700;margin-bottom:4px;color:#111827;}
.auth-subtitle{font-size:13px;color:#6b7280;margin-bottom:24px;}
.auth-switch{text-align:center;margin-top:16px;font-size:13px;color:#6b7280;}

.sidebar-user{background:linear-gradient(135deg,#4f46e5,#4338ca);color:white;padding:20px;border-radius:14px;margin-bottom:16px;}
.sidebar-user .avatar{width:48px;height:48px;border-radius:50%;background:rgba(255,255,255,0.2);display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:700;margin-bottom:12px;}
.sidebar-user .name{font-size:15px;font-weight:700;margin-bottom:2px;}
.sidebar-user .email{font-size:11px;color:rgba(255,255,255,0.8);word-break:break-all;}

.history-card{background:white;border:1px solid #e5e7eb;border-radius:12px;padding:18px 20px;margin-bottom:12px;}
.history-row{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
.history-route{font-size:15px;font-weight:700;color:#111827;}
.history-time{font-size:11px;color:#6b7280;}
.history-details{display:flex;gap:8px;flex-wrap:wrap;margin-top:8px;}
.history-chip{background:#f3f4f6;color:#374151;padding:4px 10px;border-radius:6px;font-size:11px;font-weight:600;}
.history-cost{font-size:18px;font-weight:800;color:#4f46e5;}

.empty-state{background:#f9fafb;border:2px dashed #e5e7eb;border-radius:14px;padding:40px 24px;text-align:center;}
.empty-icon{font-size:48px;margin-bottom:12px;}
.empty-title{font-size:16px;font-weight:700;color:#374151;margin-bottom:16px;}
.empty-features{text-align:left;max-width:380px;margin:0 auto;font-size:13px;color:#6b7280;line-height:1.8;}

.app-footer{text-align:center;padding:20px;font-size:11px;color:#6b7280;margin-top:32px;border-top:1px solid #f3f4f6;}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  FIREBASE
# ═══════════════════════════════════════════════════════════════════
def init_firebase():
    if firebase_admin._apps:
        return
    try:
        if "firebase_service_account" in st.secrets:
            cred_dict = dict(st.secrets["firebase_service_account"])
            cred = credentials.Certificate(cred_dict)
        else:
            cred = credentials.Certificate("firebase-service-account.json")
        firebase_admin.initialize_app(cred)
    except Exception as e:
        st.error(f"❌ Firebase init failed: {e}")


def sign_up_user(email, password, display_name):
    try:
        user = auth.create_user(email=email, password=password, display_name=display_name)
        db = firestore.client()
        db.collection("users").document(user.uid).set({
            "email": email, "display_name": display_name,
            "created_at": firestore.SERVER_TIMESTAMP,
            "prediction_count": 0
        })
        return sign_in_user(email, password)
    except auth.EmailAlreadyExistsError:
        return {"success": False, "error": "Email already registered."}
    except Exception as e:
        return {"success": False, "error": str(e)}


def sign_in_user(email, password):
    api_key = st.secrets.get("firebase_web_api_key", "")
    if not api_key:
        try:
            user = auth.get_user_by_email(email)
            st.session_state.user = {
                "uid": user.uid, "email": user.email,
                "display_name": user.display_name or email.split("@")[0]
            }
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}

    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    try:
        r = requests.post(url, json={"email": email, "password": password, "returnSecureToken": True}, timeout=10)
        if r.status_code == 200:
            data = r.json()
            st.session_state.user = {
                "uid": data["localId"], "email": data["email"],
                "display_name": data.get("displayName", email.split("@")[0])
            }
            return {"success": True}
        else:
            err = r.json().get("error", {}).get("message", "Unknown error")
            friendly = {"EMAIL_NOT_FOUND": "Email not registered.",
                        "INVALID_PASSWORD": "Incorrect password.",
                        "INVALID_LOGIN_CREDENTIALS": "Invalid email or password."}.get(err, err)
            return {"success": False, "error": friendly}
    except Exception as e:
        return {"success": False, "error": str(e)}


def save_prediction(uid, inputs, result):
    try:
        db = firestore.client()
        db.collection("predictions").add({
            "user_id": uid,
            "inputs": {
                "origin": inputs["origin"], "destination": inputs["destination"],
                "date": inputs["date"].isoformat(), "trip_type": inputs["trip_type"],
                "vehicle_cat": int(inputs["vehicle_cat"]),
                "weight": float(inputs["weight"]), "cbm": float(inputs["cbm"]),
                "eff_km": float(inputs["eff_km"]),
                "weather": {"rainfall_mm": float(inputs["weather"]["rainfall_mm"]),
                            "temp_max_c": float(inputs["weather"]["temp_max_c"]),
                            "label": inputs["weather"]["label"]}
            },
            "result": {
                "ml_cost": float(result["ml_cost"]),
                "trad_cost": float(result["trad_cost"]),
                "est_fuel": float(result["est_fuel"]),
                "delta_pct": float(result["delta_pct"]),
                "fuel_price": float(result["fuel_price"]),
                "season_label": result["season_label"],
                "model_used": result["model_used"]
            },
            "timestamp": firestore.SERVER_TIMESTAMP
        })
        db.collection("users").document(uid).update({
            "prediction_count": firestore.Increment(1),
            "last_prediction_at": firestore.SERVER_TIMESTAMP
        })
    except Exception as e:
        st.warning(f"⚠️ Could not save: {e}")


def get_user_predictions(uid, limit=50):
    try:
        db = firestore.client()
        docs = db.collection("predictions").where("user_id", "==", uid)\
            .order_by("timestamp", direction=firestore.Query.DESCENDING).limit(limit).stream()
        return [{**doc.to_dict(), "id": doc.id} for doc in docs]
    except Exception as e:
        st.warning(f"⚠️ Could not load history: {e}")
        return []


# ═══════════════════════════════════════════════════════════════════
#  ML MODEL
# ═══════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    try:
        if Path("model_b64.txt").exists():
            with open("model_b64.txt", "r") as f:
                model = pickle.loads(base64.b64decode(f.read().strip()))
        else:
            with open("rf_model.pkl", "rb") as f:
                model = pickle.load(f)
        with open("feature_cols.pkl", "rb") as f:
            cols = pickle.load(f)
        return model, cols
    except Exception:
        return None, None


# ═══════════════════════════════════════════════════════════════════
#  APIs
# ═══════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def get_road_distance(orig, dest):
    o, d = SL_CITIES[orig], SL_CITIES[dest]
    url = f"http://router.project-osrm.org/route/v1/driving/{o['lon']},{o['lat']};{d['lon']},{d['lat']}?overview=false"
    try:
        r = requests.get(url, timeout=8)
        if r.status_code == 200:
            route = r.json()["routes"][0]
            km = round(route["distance"] / 1000, 1)
            return {"one_way_km": km, "round_trip_km": round(km * 2, 1),
                    "duration_min": round(route["duration"] / 60),
                    "source": "OSRM (OpenStreetMap)"}
    except Exception:
        pass
    dlat = math.radians(d["lat"] - o["lat"])
    dlon = math.radians(d["lon"] - o["lon"])
    a = math.sin(dlat/2)**2 + math.cos(math.radians(o["lat"])) * math.cos(math.radians(d["lat"])) * math.sin(dlon/2)**2
    km = round(6371 * 2 * math.asin(math.sqrt(a)) * 1.3, 1)
    return {"one_way_km": km, "round_trip_km": round(km*2, 1),
            "duration_min": round(km / 50 * 60), "source": "Haversine estimate"}


@st.cache_data(ttl=3600)
def get_weather(city, date):
    coords = SL_CITIES[city]
    is_future = date >= datetime.date.today()
    base = "https://api.open-meteo.com/v1/forecast" if is_future else "https://archive-api.open-meteo.com/v1/archive"
    try:
        r = requests.get(base, params={
            "latitude": coords["lat"], "longitude": coords["lon"],
            "daily": "rain_sum,temperature_2m_max,precipitation_hours",
            "timezone": "Asia/Colombo",
            "start_date": str(date), "end_date": str(date)
        }, timeout=8)
        if r.status_code == 200:
            d = r.json()["daily"]
            rain = d["rain_sum"][0] or 0
            temp = d["temperature_2m_max"][0] or 28
            label = ("HEAVY RAIN ⛈️" if rain >= 25 else
                     "RAINY 🌧️" if rain >= 5 else
                     "LIGHT RAIN 🌦️" if rain >= 1 else "DRY ☀️")
            return {"rainfall_mm": round(rain, 1), "temp_max_c": round(temp, 1),
                    "rain_hours": round(d["precipitation_hours"][0] or 0, 1),
                    "label": label, "source": "Forecast" if is_future else "Archive"}
    except Exception:
        pass
    return {"rainfall_mm": 0, "temp_max_c": 28, "rain_hours": 0,
            "label": "UNKNOWN", "source": "Fallback"}


# ═══════════════════════════════════════════════════════════════════
#  PREDICTION LOGIC
# ═══════════════════════════════════════════════════════════════════
def predict_cost(inputs, rf_model, feature_cols):
    eff_km = inputs["eff_km"]
    vcat = inputs["vehicle_cat"]
    weather = inputs["weather"]
    date = inputs["date"]
    weight = inputs["weight"]
    cbm = inputs["cbm"]
    city = inputs["destination"]

    fuel_price = FUEL_PRICES.get(f"{date.year}-{date.month:02d}", 320)
    fuel_cons = FUEL_CONS[vcat]
    est_fuel = (eff_km / 100) * fuel_cons * fuel_price
    season = 1 if date.month in [12,1,2] else 2 if date.month in [5,6,7,8,9] else 0
    trad_cost = eff_km * KM_RATES[vcat]

    features = {
        "approved_km": eff_km, "weight": weight, "cbm": cbm, "vehicle_cat": vcat,
        "month": date.month, "quarter": (date.month - 1) // 3 + 1,
        "day_of_week": date.weekday(),
        "is_weekend": 1 if date.weekday() >= 5 else 0,
        "season": season, "is_monsoon": 1 if season > 0 else 0,
        "rainfall_mm": weather["rainfall_mm"], "temp_max_c": weather["temp_max_c"],
        "rain_hours": weather["rain_hours"],
        "is_rainy_day": 1 if weather["rainfall_mm"] >= 5 else 0,
        "is_heavy_rain": 1 if weather["rainfall_mm"] >= 25 else 0,
        "is_hot_day": 1 if weather["temp_max_c"] >= 33 else 0,
        "fuel_consumption_per100": fuel_cons, "est_fuel_cost": est_fuel,
        "vh_cost_per_km": VH_AVG_COST[vcat] / AVG_KM,
        "fuel_cost_ratio": min(est_fuel / max(trad_cost, 1), 1.0),
        "is_urban": 1 if city in URBAN_CITIES else 0,
        "road_complexity": 3 if eff_km < 50 else 2 if eff_km < 150 else 1 if eff_km < 500 else 0,
        "is_peak_day": 1 if date.weekday() < 5 else 0,
        "fuel_price_lkr": fuel_price, "distance_sq": eff_km ** 2,
        "weight_distance": (weight * eff_km) / 1000,
        "km_gap": 0, "km_gap_pct": 0,
    }

    season_labels = {0: "Inter-monsoon", 1: "NE Monsoon (Dec-Feb)", 2: "SW Monsoon (May-Sep)"}

    if rf_model is not None and feature_cols is not None:
        X = pd.DataFrame([[features.get(c, 0) for c in feature_cols]], columns=feature_cols)
        ml_cost = float(rf_model.predict(X)[0])
        model_used = "🤖 Random Forest model (R²=0.9961, MAPE=3.54%)"
    else:
        ml_cost = trad_cost * 1.18
        model_used = "📊 Formula-based estimate (model not loaded)"

    return {
        "ml_cost": round(ml_cost), "trad_cost": round(trad_cost),
        "est_fuel": round(est_fuel), "eff_km": eff_km,
        "fuel_price": fuel_price,
        "delta_pct": round((ml_cost - trad_cost) / max(trad_cost, 1) * 100, 1),
        "model_used": model_used, "season_label": season_labels[season],
    }


# ═══════════════════════════════════════════════════════════════════
#  UI COMPONENTS
# ═══════════════════════════════════════════════════════════════════
def render_metric(label, value, delta, primary=False):
    cls = "primary" if primary else ""
    st.markdown(
        f'<div class="metric-card {cls}"><div class="metric-label">{label}</div>'
        f'<div class="metric-value {cls}">{value}</div>'
        f'<div class="metric-delta">{delta}</div></div>',
        unsafe_allow_html=True
    )


def render_factor(label, value):
    st.markdown(
        f'<div class="factor-row"><span class="factor-label">{label}</span>'
        f'<span class="factor-value">{value}</span></div>',
        unsafe_allow_html=True
    )


def render_login():
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="auth-logo"><span class="icon">🚛</span>'
        '<h1>FreightIQ Pro</h1><p>Real-time freight cost prediction</p></div>'
        '<div class="auth-title">Welcome back</div>'
        '<div class="auth-subtitle">Sign in to access your predictions</div>',
        unsafe_allow_html=True
    )
    email = st.text_input("Email", placeholder="you@example.com", key="li_email")
    pwd = st.text_input("Password", type="password", placeholder="••••••••", key="li_pwd")
    if st.button("Sign in", type="primary", use_container_width=True, key="li_btn"):
        if not email or not pwd:
            st.error("Please enter both email and password.")
        else:
            r = sign_in_user(email, pwd)
            if r["success"]:
                st.session_state.page = "predict"
                st.rerun()
            else:
                st.error(f"❌ {r['error']}")
    st.markdown('<div class="auth-switch">Don\'t have an account?</div>', unsafe_allow_html=True)
    if st.button("Create account", use_container_width=True, key="li_signup"):
        st.session_state.page = "signup"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_signup():
    st.markdown('<div class="auth-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="auth-logo"><span class="icon">🚛</span>'
        '<h1>FreightIQ Pro</h1><p>Real-time freight cost prediction</p></div>'
        '<div class="auth-title">Create your account</div>'
        '<div class="auth-subtitle">Save predictions, view history</div>',
        unsafe_allow_html=True
    )
    name = st.text_input("Full name", placeholder="Ridma Sanka", key="su_name")
    email = st.text_input("Email", placeholder="you@example.com", key="su_email")
    pwd = st.text_input("Password (min 6 chars)", type="password", placeholder="••••••••", key="su_pwd")
    if st.button("Create account", type="primary", use_container_width=True, key="su_btn"):
        if not all([name, email, pwd]):
            st.error("All fields required.")
        elif len(pwd) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            r = sign_up_user(email, pwd, name)
            if r["success"]:
                st.session_state.page = "predict"
                st.rerun()
            else:
                st.error(f"❌ {r['error']}")
    st.markdown('<div class="auth-switch">Already have an account?</div>', unsafe_allow_html=True)
    if st.button("Sign in instead", use_container_width=True, key="su_login"):
        st.session_state.page = "login"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
#  PAGES
# ═══════════════════════════════════════════════════════════════════
def page_predict(rf_model, feature_cols):
    st.markdown(
        '<div class="app-header"><h1>🚛 Real-Time Cost Prediction</h1>'
        '<p>Multi-factor freight cost estimation powered by ML</p></div>',
        unsafe_allow_html=True
    )
    col_l, col_r = st.columns([1, 1.4], gap="large")

    with col_l:
        st.markdown('<div class="card"><div class="card-title">🚚 Trip Details</div>', unsafe_allow_html=True)
        cities = sorted(SL_CITIES.keys())
        origin = st.selectbox("📍 Origin", cities, index=cities.index("Colombo"))
        dest = st.selectbox("🎯 Destination", cities, index=cities.index("Kandy"))
        ca, cb = st.columns(2)
        with ca:
            trip_date = st.date_input("📅 Trip Date",
                value=datetime.date.today(),
                min_value=datetime.date(2023, 1, 1),
                max_value=datetime.date.today() + datetime.timedelta(days=16))
        with cb:
            trip_type = st.selectbox("🔄 Trip Type", ["One-way", "Round-trip"])
        vehicle = st.selectbox("🚛 Vehicle Type", [0, 1, 2],
                               format_func=lambda x: VH_NAMES[x], index=1)
        cw, cv = st.columns(2)
        with cw:
            weight = st.number_input("⚖️ Weight (kg)", min_value=0.0, max_value=30000.0, value=2000.0, step=100.0)
        with cv:
            cbm = st.number_input("📦 Volume (CBM)", min_value=0.0, max_value=80.0, value=8.0, step=0.5)
        predict_btn = st.button("⚡ Predict Cost", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_r:
        if predict_btn:
            with st.spinner("🛰️ Fetching distance & weather..."):
                dist = get_road_distance(origin, dest)
                weather = get_weather(dest, trip_date)
                eff_km = dist["round_trip_km"] if trip_type == "Round-trip" else dist["one_way_km"]
                inputs = {"origin": origin, "destination": dest, "date": trip_date,
                          "trip_type": trip_type, "vehicle_cat": vehicle,
                          "weight": weight, "cbm": cbm, "eff_km": eff_km,
                          "weather": weather, "dist_data": dist}
                result = predict_cost(inputs, rf_model, feature_cols)
                st.session_state.last_prediction = {"inputs": inputs, "result": result}
                if st.session_state.user:
                    save_prediction(st.session_state.user["uid"], inputs, result)

        if st.session_state.last_prediction:
            p = st.session_state.last_prediction
            inp = p["inputs"]
            res = p["result"]
            w = inp["weather"]
            d = inp["dist_data"]
            wl = w["label"]
            badge = "badge-heavy" if "HEAVY" in wl else "badge-rainy" if "RAIN" in wl else "badge-dry"

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-header"><span class="result-route">{inp["origin"]} → {inp["destination"]}</span>'
                f'<span class="{badge}">{wl}</span></div>'
                f'<div class="result-meta">{w["rainfall_mm"]}mm · {w["temp_max_c"]}°C · {w["source"]}</div>',
                unsafe_allow_html=True
            )
            c1, c2, c3 = st.columns(3)
            sign = "+" if res["delta_pct"] >= 0 else ""
            with c1:
                render_metric("ML Prediction", f"LKR {res['ml_cost']:,.0f}",
                              f"{sign}{res['delta_pct']}% vs flat rate", primary=True)
            with c2:
                render_metric("Traditional", f"LKR {res['trad_cost']:,.0f}",
                              f"LKR {KM_RATES[inp['vehicle_cat']]}/km")
            with c3:
                render_metric("Est. Fuel", f"LKR {res['est_fuel']:,.0f}",
                              f"{FUEL_CONS[inp['vehicle_cat']]}L/100km")

            st.markdown('<div class="section-title">Cost Comparison</div>', unsafe_allow_html=True)
            st.bar_chart(pd.DataFrame({
                "Method": ["ML Prediction", "Traditional", "Est. Fuel"],
                "LKR": [res["ml_cost"], res["trad_cost"], res["est_fuel"]]
            }).set_index("Method"), height=220, color="#6366f1")

            st.markdown('<div class="section-title">Trip Factors</div>', unsafe_allow_html=True)
            for label, value in {
                "🛣️ Road distance": f"{d['one_way_km']} km one-way · {d['round_trip_km']} km round-trip",
                "⏱️ Duration": f"~{d['duration_min']} min",
                "📏 Effective KM": f"{res['eff_km']} km ({inp['trip_type']})",
                "🚛 Vehicle": VH_NAMES[inp["vehicle_cat"]],
                "🌦️ Season": res["season_label"],
                "⛽ Fuel price": f"LKR {res['fuel_price']}/L",
                "🗺️ Distance API": d["source"],
            }.items():
                render_factor(label, value)

            st.markdown(f'<div class="model-tag">{res["model_used"]}</div>', unsafe_allow_html=True)
            if st.session_state.user:
                st.success("✅ Prediction saved to your history")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            features = [
                "🛣️ Real road distance via OSRM (OpenStreetMap)",
                "🌦️ Weather forecast for future trips (up to 16 days)",
                "📅 Historical weather for past dates",
                "🤖 Random Forest model (R²=0.9961, MAPE=3.54%)",
                "🔄 Round-trip cost support",
                "💾 Auto-save to your prediction history"
            ]
            features_html = "".join(f"<div>• {f}</div>" for f in features)
            st.markdown(
                f'<div class="empty-state"><div class="empty-icon">🚚</div>'
                f'<div class="empty-title">👈 Fill in trip details and click Predict Cost</div>'
                f'<div class="empty-features">{features_html}</div></div>',
                unsafe_allow_html=True
            )


def page_history():
    st.markdown(
        '<div class="app-header"><h1>📊 Prediction History & Analytics</h1>'
        '<p>Your past freight cost predictions with data visualization</p></div>',
        unsafe_allow_html=True
    )
    if not st.session_state.user:
        st.warning("Please log in.")
        return

    preds = get_user_predictions(st.session_state.user["uid"])
    if not preds:
        st.markdown(
            '<div class="empty-state"><div class="empty-icon">📭</div>'
            '<div class="empty-title">No predictions yet</div>'
            '<div class="empty-features">Make your first prediction to see it here.<br>'
            'All predictions auto-save when you\'re logged in.</div></div>',
            unsafe_allow_html=True
        )
        return

    # ─── SUMMARY METRICS ──────────────────────────────────────────
    total = len(preds)
    total_val = sum(p["result"]["ml_cost"] for p in preds)
    avg_delta = sum(p["result"]["delta_pct"] for p in preds) / total if total else 0
    avg_cost = total_val / total if total else 0

    c1, c2, c3, c4 = st.columns(4)
    with c1: render_metric("Total Predictions", f"{total}", "all-time", primary=True)
    with c2: render_metric("Total Value", f"LKR {total_val:,.0f}", "sum of ML predictions")
    with c3: render_metric("Avg Cost", f"LKR {avg_cost:,.0f}", "per prediction")
    with c4: render_metric("Avg Adjustment", f"{avg_delta:+.1f}%", "vs flat rate")

    # ─── BUILD DATAFRAME for charts ───────────────────────────────
    df_data = []
    for p in preds:
        inp = p["inputs"]
        res = p["result"]
        ts = p.get("timestamp")
        try:
            dt = ts if hasattr(ts, 'year') else None
        except Exception:
            dt = None

        df_data.append({
            "route": f"{inp['origin']} → {inp['destination']}",
            "origin": inp["origin"],
            "destination": inp["destination"],
            "ml_cost": res["ml_cost"],
            "trad_cost": res["trad_cost"],
            "delta_pct": res["delta_pct"],
            "vehicle": VH_NAMES.get(inp.get("vehicle_cat", 1), "—"),
            "trip_date": inp.get("date", ""),
            "timestamp": dt,
            "eff_km": inp.get("eff_km", 0),
            "weight": inp.get("weight", 0),
            "weather": inp.get("weather", {}).get("label", "—"),
            "trip_type": inp.get("trip_type", "—"),
        })

    df = pd.DataFrame(df_data)

    # ─── COMMON CHART STYLING ─────────────────────────────────────
    chart_layout = dict(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="-apple-system, sans-serif", size=12, color="#374151"),
        margin=dict(l=20, r=20, t=40, b=20),
        height=320,
    )
    primary_color = "#6366f1"
    color_palette = ["#6366f1", "#8b5cf6", "#a855f7", "#d946ef", "#ec4899"]

    # ═══════════════════════════════════════════════════════════════
    #  ANALYTICS SECTION
    # ═══════════════════════════════════════════════════════════════
    st.markdown('<div class="section-title">📈 Analytics Dashboard</div>',
                unsafe_allow_html=True)

    # ─── CHART 1 & 2: Cost trend + Vehicle distribution (side by side) ──
    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📊 Cost Trend Over Time</div>',
                    unsafe_allow_html=True)

        # Sort chronologically (oldest → newest)
        df_sorted = df.sort_values('timestamp', na_position='first').reset_index(drop=True)
        df_sorted["prediction_no"] = range(1, len(df_sorted) + 1)

        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df_sorted["prediction_no"],
            y=df_sorted["ml_cost"],
            mode='lines+markers',
            name='ML Prediction',
            line=dict(color=primary_color, width=3),
            marker=dict(size=8, color=primary_color,
                        line=dict(width=2, color='white')),
            hovertemplate='<b>Prediction #%{x}</b><br>'
                          'Cost: LKR %{y:,.0f}<br>'
                          'Route: %{customdata[0]}<br>'
                          'Vehicle: %{customdata[1]}<extra></extra>',
            customdata=df_sorted[["route", "vehicle"]].values,
            fill='tozeroy',
            fillcolor='rgba(99,102,241,0.1)'
        ))
        fig1.update_layout(
            **chart_layout,
            xaxis=dict(title="Prediction Sequence", gridcolor="#f3f4f6", showgrid=True),
            yaxis=dict(title="ML Cost (LKR)", gridcolor="#f3f4f6", showgrid=True),
            showlegend=False
        )
        st.plotly_chart(fig1, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">🚛 Vehicle Type Distribution</div>',
                    unsafe_allow_html=True)

        vehicle_counts = df["vehicle"].value_counts()
        fig2 = go.Figure(data=[go.Pie(
            labels=vehicle_counts.index,
            values=vehicle_counts.values,
            hole=0.55,
            marker=dict(colors=color_palette[:len(vehicle_counts)],
                        line=dict(color='white', width=3)),
            textfont=dict(size=14, color='white'),
            textposition='inside',
            textinfo='percent',
            hovertemplate='<b>%{label}</b><br>'
                          'Count: %{value}<br>'
                          'Percentage: %{percent}<extra></extra>'
        )])
        fig2.update_layout(
            **chart_layout,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.15,
                        xanchor="center", x=0.5),
            annotations=[dict(text=f'<b>{total}</b><br>trips',
                              x=0.5, y=0.5, font_size=18, showarrow=False,
                              font_color="#111827")]
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ─── CHART 3 & 4: Top routes + Monthly trend ──────────────────
    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📍 Top Routes</div>',
                    unsafe_allow_html=True)

        route_counts = df["route"].value_counts().head(8)
        fig3 = go.Figure(go.Bar(
            x=route_counts.values,
            y=route_counts.index,
            orientation='h',
            marker=dict(
                color=route_counts.values,
                colorscale=[[0, "#c7d2fe"], [1, "#4f46e5"]],
                showscale=False,
                line=dict(color='white', width=1)
            ),
            text=route_counts.values,
            textposition='outside',
            textfont=dict(size=12, color="#374151"),
            hovertemplate='<b>%{y}</b><br>Predictions: %{x}<extra></extra>'
        ))
        fig3.update_layout(
            **chart_layout,
            xaxis=dict(title="Number of Predictions", gridcolor="#f3f4f6"),
            yaxis=dict(title="", autorange="reversed"),
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    with chart_col4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-title">📅 Monthly Predictions</div>',
                    unsafe_allow_html=True)

        # Group by month
        df_monthly = df.dropna(subset=['timestamp']).copy()
        if not df_monthly.empty:
            df_monthly['month'] = pd.to_datetime(df_monthly['timestamp']).dt.strftime('%Y-%m')
            monthly_counts = df_monthly.groupby('month').size().reset_index(name='count')
            monthly_costs = df_monthly.groupby('month')['ml_cost'].sum().reset_index()

            fig4 = go.Figure()
            fig4.add_trace(go.Bar(
                x=monthly_counts['month'],
                y=monthly_counts['count'],
                name='Predictions',
                marker=dict(color=primary_color,
                            line=dict(color='white', width=1)),
                hovertemplate='<b>%{x}</b><br>Predictions: %{y}<extra></extra>',
                yaxis='y'
            ))
            fig4.add_trace(go.Scatter(
                x=monthly_costs['month'],
                y=monthly_costs['ml_cost'],
                mode='lines+markers',
                name='Total Cost',
                line=dict(color='#ec4899', width=3),
                marker=dict(size=10, color='#ec4899',
                            line=dict(width=2, color='white')),
                yaxis='y2',
                hovertemplate='<b>%{x}</b><br>Total: LKR %{y:,.0f}<extra></extra>'
            ))
            fig4.update_layout(
                **chart_layout,
                xaxis=dict(title="Month", gridcolor="#f3f4f6"),
                yaxis=dict(title="Predictions", gridcolor="#f3f4f6", side='left'),
                yaxis2=dict(title="Total Cost (LKR)", overlaying='y',
                            side='right', showgrid=False),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            xanchor="center", x=0.5),
                showlegend=True
            )
            st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
        else:
            st.markdown(
                '<div style="text-align:center; padding:40px; color:#6b7280;">'
                'Monthly data will appear after predictions accumulate over time.'
                '</div>',
                unsafe_allow_html=True
            )
        st.markdown('</div>', unsafe_allow_html=True)

    # ─── DETAILED LIST: Recent Predictions ────────────────────────
    st.markdown('<div class="section-title">📋 Recent Predictions</div>',
                unsafe_allow_html=True)

    for p in preds:
        inp = p["inputs"]
        res = p["result"]
        ts = p.get("timestamp")
        try:
            time_str = ts.strftime("%b %d, %Y · %I:%M %p")
        except (AttributeError, TypeError):
            time_str = "Recent"
        vh = VH_NAMES.get(inp.get("vehicle_cat", 1), "—")
        wl = inp.get("weather", {}).get("label", "—")

        st.markdown(
            f'<div class="history-card"><div class="history-row">'
            f'<div><div class="history-route">{inp["origin"]} → {inp["destination"]}</div>'
            f'<div class="history-time">{time_str} · {inp.get("date", "")}</div></div>'
            f'<div style="text-align:right"><div class="history-cost">LKR {res["ml_cost"]:,.0f}</div>'
            f'<div class="history-time">{res["delta_pct"]:+.1f}% vs flat</div></div></div>'
            f'<div class="history-details">'
            f'<span class="history-chip">🚛 {vh}</span>'
            f'<span class="history-chip">📏 {inp["eff_km"]:.0f} km</span>'
            f'<span class="history-chip">⚖️ {inp["weight"]:.0f} kg</span>'
            f'<span class="history-chip">{wl}</span>'
            f'<span class="history-chip">{inp.get("trip_type", "—")}</span>'
            f'</div></div>',
            unsafe_allow_html=True
        )


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════
def main():
    init_firebase()

    if "user" not in st.session_state: st.session_state.user = None
    if "page" not in st.session_state: st.session_state.page = "login"
    if "last_prediction" not in st.session_state: st.session_state.last_prediction = None

    rf_model, feature_cols = load_model()

    if not st.session_state.user:
        if st.session_state.page == "signup":
            render_signup()
        else:
            render_login()
        st.markdown('<div class="app-footer">FreightIQ Pro · Thesis ST87628 · TTI Riga · 2026</div>',
                    unsafe_allow_html=True)
        return

    with st.sidebar:
        user = st.session_state.user
        initial = (user.get("display_name") or user.get("email", "U"))[0].upper()
        st.markdown(
            f'<div class="sidebar-user"><div class="avatar">{initial}</div>'
            f'<div class="name">{user.get("display_name", "User")}</div>'
            f'<div class="email">{user["email"]}</div></div>',
            unsafe_allow_html=True
        )
        st.markdown("---")
        page = st.radio("Navigation", ["🚛 New Prediction", "📊 History"],
                        label_visibility="collapsed")
        st.markdown("---")
        if st.button("🚪 Sign Out", use_container_width=True):
            st.session_state.user = None
            st.session_state.last_prediction = None
            st.session_state.page = "login"
            st.rerun()

    if "New Prediction" in page:
        page_predict(rf_model, feature_cols)
    else:
        page_history()

    st.markdown('<div class="app-footer">FreightIQ Pro · Thesis ST87628 · TTI Riga · 2026 · @Sanka-96</div>',
                unsafe_allow_html=True)


if __name__ == "__main__":
    main()
