import os, glob, json, io, time
from typing import List, Tuple, Dict, Any
import numpy as np
from PIL import Image
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models
from shutil import copyfile
import csv, time, os
import streamlit as st
import pandas as pd
import plotly.express as px

# --- NEW IMPORTS ---
import requests
import google.generativeai as genai
from dotenv import load_dotenv
import re  # for JSON repair
# ---------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_PREFIX = "soil_memory"  # soil_memory_X.npy + soil_memory_meta.json

# --- Gemini API Configuration ---
load_dotenv()
api_key = os.environ.get("GOOGLE_API_KEY")

if not api_key:
    st.error("API key not found. Please create a .env file with GOOGLE_API_KEY.")
else:
    genai.configure(api_key=api_key)

# --- Gemini Model Caching ---
@st.cache_resource(show_spinner="Loading Gemini AI model...")
def get_gemini_model():
    """
    Returns a GenerativeModel your key can actually use.
    Prefers 2.5/`latest` models; falls back gracefully.
    """
    available = []
    try:
        for m in genai.list_models():
            if "generateContent" in getattr(m, "supported_generation_methods", []):
                available.append(m.name)  # e.g., "models/gemini-2.5-flash"
    except Exception:
        pass

    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.5-pro",
        "models/gemini-flash-latest",
        "models/gemini-pro-latest",
        "models/gemini-2.0-flash",
        "models/gemini-2.0-flash-001",
        "models/gemini-1.5-flash",
        "models/gemini-1.5-pro",
        "models/gemini-pro",
    ]
    chosen = next((m for m in preferred if m in available), None)

    candidates = [chosen] if chosen else (available or preferred)
    for c in candidates:
        try:
            mdl = genai.GenerativeModel(c)
            # Store model name in session state for sidebar display
            if 'gemini_model_name' not in st.session_state:
                st.session_state.gemini_model_name = c
            return mdl
        except Exception:
            continue

    raise RuntimeError(f"No compatible Gemini model found. Visible: {available or '(none)'}")

# --- JSON Repair/Parser ---
def parse_llm_json_output(raw_output: str):
    """
    Attempts to coerce almost-JSON from LLMs into valid JSON:
    - strips code fences
    - normalizes quotes
    - trims to the outermost { ... }
    - removes trailing commas
    - balances braces/brackets if off by a little
    """
    try:
        s = (raw_output or "").strip()

        # strip ``` and ```json fences
        s = re.sub(r"```(?:json)?", "", s, flags=re.IGNORECASE).replace("```", "").strip()

        # normalize smart quotes -> straight quotes
        s = (s.replace("“", '"').replace("”", '"')
               .replace("’", "'").replace("‘", "'"))

        # keep only the outermost JSON object
        start, end = s.find("{"), s.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object braces found.")
        s = s[start:end+1]

        # remove trailing commas before } or ]
        s = re.sub(r",\s*(?=[}\]])", "", s)

        # balance braces/brackets if off by a small count
        diff_brace = s.count("{") - s.count("}")
        if diff_brace > 0:
            s += "}" * diff_brace
        diff_brack = s.count("[") - s.count("]")
        if diff_brack > 0:
            s += "]" * diff_brack

        return json.loads(s)
    except Exception as e:
        st.error(f"Error decoding AI JSON response: {e}")
        st.text_area("Raw AI Output", raw_output, height=240)
        return None
# ---------------------------------


# ---------- Model / Featurizer ----------
@st.cache_resource(show_spinner="Loading classification model...")
def get_featurizer():
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(m.children())[:-1]).to(DEVICE).eval()
    return backbone

PREPROCESS = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

def embed_pil(img: Image.Image, backbone) -> np.ndarray:
    img = img.convert("RGB")
    x = PREPROCESS(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        f = backbone(x).squeeze(-1).squeeze(-1)
        f = nn.functional.normalize(f, dim=-1)
    return f.cpu().numpy().astype(np.float32)

def embed_path(path: str, backbone) -> np.ndarray:
    return embed_pil(Image.open(path), backbone)

# ---------- Index build/load ----------
def list_classes(data_dir: str) -> List[str]:
    return sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

def safe_relpath(p: str) -> str:
    return os.path.relpath(p, start=os.getcwd())

def build_index(data_dir: str, prefix: str = INDEX_PREFIX) -> Tuple[np.ndarray, dict]:
    backbone = get_featurizer()
    classes = list_classes(data_dir)
    labels, paths, vecs = [], [], []
    for c in classes:
        for fp in glob.glob(os.path.join(data_dir, c, "*")):
            if os.path.isdir(fp):
                continue
            ext = os.path.splitext(fp)[1].lower()
            if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
                continue
            try:
                v = embed_path(fp, backbone)
                labels.append(c); paths.append(safe_relpath(fp)); vecs.append(v)
            except Exception:
                pass
    if len(vecs) == 0:
        raise RuntimeError("No images found to build the index.")
    X = np.vstack([v.reshape(1, -1) for v in vecs]).astype(np.float32)
    X = normalize(X)
    np.save(prefix + "_X.npy", X)
    meta = {"labels": labels, "paths": paths, "classes": classes, "data_dir": data_dir}
    with open(prefix + "_meta.json", "w") as f:
        json.dump(meta, f)
    return X, meta

def load_index(prefix: str = INDEX_PREFIX) -> Tuple[np.ndarray, dict]:
    X = np.load(prefix + "_X.npy")
    with open(prefix + "_meta.json") as f:
        meta = json.load(f)
    return X, meta

def ensure_index(data_dir: str):
    if not (os.path.exists(INDEX_PREFIX + "_X.npy") and os.path.exists(INDEX_PREFIX + "_meta.json")):
        return build_index(data_dir)
    return load_index()

def log_prediction(logfile, src, shown_label, top_label, sim, low_conf):
    os.makedirs(os.path.dirname(logfile), exist_ok=True) if os.path.dirname(logfile) else None
    header = ["ts","source","shown_label","top_label","similarity","low_confidence"]
    ts = int(time.time() * 1000)
    write_header = not os.path.exists(logfile)
    with open(logfile, "a", newline="") as f:
        w = csv.writer(f)
        if write_header: w.writerow(header)
        w.writerow([ts, src, shown_label, top_label, f"{sim:.4f}", int(low_conf)])

# ---------- kNN Predict ----------
def knn_predict(img: Image.Image, X: np.ndarray, labels: List[str], k: int = 3):
    nbrs = NearestNeighbors(n_neighbors=min(k, len(labels)), metric="cosine").fit(X)
    q = embed_pil(img, get_featurizer()).reshape(1, -1)
    dists, idxs = nbrs.kneighbors(q, return_distance=True)
    sims = 1 - dists[0]
    return idxs[0], sims

# ---------- Utils ----------
def save_uploaded_to_class(upload, class_name: str, data_dir: str) -> str:
    os.makedirs(os.path.join(data_dir, class_name), exist_ok=True)
    ts = int(time.time() * 1000)
    ext = os.path.splitext(upload.name)[1] or ".jpg"
    fp = os.path.join(data_dir, class_name, f"user_{ts}{ext}")
    with open(fp, "wb") as f:
        f.write(upload.getbuffer())
    return fp

# --- Weather Function (Open-Meteo Climate) ---
@st.cache_data(ttl=3600)
def get_climate_data(lat, lon):
    try:
        url = "https://climate-api.open-meteo.com/v1/climate"
        params = {
            "latitude": float(lat),
            "longitude": float(lon),
            "start_date": "2010-01-01",
            "end_date": "2020-12-31",
            "daily": "temperature_2m_mean,precipitation_sum",
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Weather API Error: {e}")
        return None
    except ValueError:
        st.error("Invalid Latitude/Longitude. Please enter numbers.")
        return None

# --- UI Rendering Functions ---
def render_recommendation_cards(parsed_json: Dict[str, Any]):
    """Render AI recommendations with high-visibility UI"""

    # === High-visibility CSS (cards stay white/black regardless of global theme) ===
    st.markdown("""
    <style>
      .ai-section-title {
        font-size: 28px;
        line-height: 1.2;
        font-weight: 800;
        letter-spacing: -0.2px;
        margin: 6px 0 14px 0;
        color: #fff !important;
      }
      .sub-kicker {
        font-size: 13px;
        letter-spacing: .12em;
        text-transform: uppercase;
        opacity: .9;
        color: #cfd8dc !important;
        margin-bottom: 6px;
      }

      .recommendation-card {
        background: #ffffff !important;
        border-radius: 14px;
        padding: 22px;
        margin: 14px 0 18px;
        box-shadow: 0 10px 24px rgba(0,0,0,.08);
        border: 1px solid #e9eef3;
      }
      .recommendation-card * {
        color: #111 !important;
      }

      .card-header-row{
        display:flex;justify-content:space-between;align-items:center;gap:12px;margin-bottom:8px
      }
      .chip {
        display:inline-flex;align-items:center;gap:6px;
        padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;
        background:#eef2ff;color:#3949ab;border:1px solid #e0e7ff;
      }
      .score-bubble {
        min-width:64px;text-align:center;font-weight:800;font-size:18px;
        padding:8px 10px;border-radius:10px;background:#e8f5e9;color:#1b5e20;border:1px solid #c8e6c9;
      }
      .metric-row{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 2px}
      .metric-badge {
        display:inline-flex;align-items:center;gap:6px;
        background:#f4f6f8;border:1px solid #e5eaef;color:#222 !important;
        padding:6px 10px;border-radius:999px;font-size:12px;font-weight:600;
      }
      .metric-badge.water-low  { background:#e8f5e9;border-color:#c8e6c9;color:#1b5e20 !important; }
      .metric-badge.water-med  { background:#fff7e6;border-color:#ffe0b2;color:#e65100 !important; }
      .metric-badge.water-high { background:#e3f2fd;border-color:#bbdefb;color:#0d47a1 !important; }

      .reason-block {margin:10px 0 0}
      .reason-block ul {margin:6px 0 0 18px}
      .reason-block li {margin:4px 0}

      .cost-wrap{margin-top:8px;border-top:1px dashed #e5eaef;padding-top:8px}
      .cost-line{display:flex;justify-content:space-between;margin:4px 0;font-size:14px}
      .cost-total{font-weight:800;font-size:16px;margin-top:6px}

      .timeline{margin-top:10px}
      .pill {
        display:inline-block;margin:4px 6px 0 0;padding:6px 12px;border-radius:999px;
        font-size:12px;font-weight:700;border:1px solid #e0e3e7;background:#f7f9fb;color:#222 !important;
      }
      .pill--action { background:#4caf50;color:#fff !important;border-color:#43a047; }

      .top-plan {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 14px; padding: 22px; color: #fff; margin: 18px 0 4px;
        box-shadow: 0 10px 24px rgba(0,0,0,.10);
        border: 1px solid rgba(255,255,255,.2);
      }
      .top-plan h2, .top-plan h3, .top-plan p, .top-plan strong { color: #fff !important; }

      /* Streamlit progress bar height bump */
      .stProgress > div > div {
        height: 10px;
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sub-kicker">AI Agronomy</div>', unsafe_allow_html=True)
    st.markdown('<div class="ai-section-title">🌾 Crop Recommendations</div>', unsafe_allow_html=True)

    recommendations = parsed_json.get("ai_recommendations", [])
    top_plan = parsed_json.get("top_plan", {})
    ai_validation = parsed_json.get("ai_soil_validation")

    # --- Recommendations ---
    for i, rec in enumerate(recommendations, 1):
        crop_name   = rec.get("crop", "Unknown Crop")
        reason_raw  = rec.get("reason", "")
        score       = int(rec.get("score", 85))
        yield_val   = rec.get("yield", "N/A")
        water_req   = (rec.get("water_requirement", "Medium") or "Medium").strip().lower()
        npk         = rec.get("npk", "N/A")
        seed_rate   = rec.get("seed_rate", "N/A")
        costs       = rec.get("costs", {}) or {}
        timeline    = rec.get("timeline", {}) or {}

        # water color class
        water_cls = "water-med"
        if "low" in water_req:  water_cls = "water-low"
        if "high" in water_req: water_cls = "water-high"

        # Reason → bullet points (split on . or ; safely)
        bullets = [b.strip() for b in re.split(r"[.;]\s+", reason_raw) if b.strip()]

        st.markdown('<div class="recommendation-card">', unsafe_allow_html=True)

        # header row
        st.markdown(
            f"""
            <div class="card-header-row">
              <div class="chip">#{i} Recommendation</div>
              <div class="score-bubble">{score}/100</div>
            </div>
            <h3 style="margin:0 0 6px 0">{crop_name}</h3>
            """,
            unsafe_allow_html=True
        )

        # score bar (0–1 value)
        st.progress(max(0, min(score, 100)) / 100.0)

        # metrics row
        st.markdown('<div class="metric-row">', unsafe_allow_html=True)
        st.markdown(f'<span class="metric-badge {water_cls}">💧 Water: {rec.get("water_requirement","Medium")}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="metric-badge">📊 Yield: {yield_val}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="metric-badge">🧪 NPK: {npk}</span>', unsafe_allow_html=True)
        st.markdown(f'<span class="metric-badge">🌱 Seed: {seed_rate}</span>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # reasons
        if bullets:
            st.markdown('<div class="reason-block"><strong>Why this works:</strong><ul>', unsafe_allow_html=True)
            for b in bullets:
                st.markdown(f"<li>{b}</li>", unsafe_allow_html=True)
            st.markdown('</ul></div>', unsafe_allow_html=True)

        # costs
        seed_c = costs.get("seed", 0); fert_c = costs.get("fertilizer", 0); other_c = costs.get("other", 0)
        total_c = costs.get("total", seed_c + fert_c + other_c)
        st.markdown('<div class="cost-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="cost-line"><span>Seed</span><strong>${:,.0f}</strong></div>'.format(seed_c), unsafe_allow_html=True)
        st.markdown('<div class="cost-line"><span>Fertilizer</span><strong>${:,.0f}</strong></div>'.format(fert_c), unsafe_allow_html=True)
        if other_c:
            st.markdown('<div class="cost-line"><span>Other</span><strong>${:,.0f}</strong></div>'.format(other_c), unsafe_allow_html=True)
        st.markdown('<div class="cost-total">Total: ${:,.0f}</div>'.format(total_c), unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # timeline
        planting = timeline.get("planting_window", "")
        harvest  = timeline.get("harvest_window", "")
        if planting or harvest:
            st.markdown('<div class="timeline"><strong>Timeline:</strong><br>', unsafe_allow_html=True)
            if planting:
                # split "Apr-May" or "Apr, May"
                pmonths = re.split(r'[,\-]', planting) if re.search(r'[,\-]', planting) else [planting]
                for m in [mm.strip() for mm in pmonths if mm.strip()]:
                    st.markdown(f'<span class="pill">Plant {m}</span>', unsafe_allow_html=True)
            if harvest:
                hmonths = re.split(r'[,\-]', harvest) if re.search(r'[,\-]', harvest) else [harvest]
                for m in [mm.strip() for mm in hmonths if mm.strip()]:
                    st.markdown(f'<span class="pill pill--action">Harvest {m}</span>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)  # /recommendation-card

    # --- Top Plan (high contrast banner) ---
    if top_plan:
        crop_name = top_plan.get("crop_name", "Unknown")
        seed_rate = top_plan.get("seed_rate_per_acre", "N/A")
        npk       = top_plan.get("fertilizer_npk_per_acre", "N/A")
        costs     = top_plan.get("est_cost_cad_per_acre", {}) or {}
        sched     = top_plan.get("schedule", {}) or {}
        insights  = top_plan.get("ai_insights", "")

        seed_c = costs.get("seed", 0); fert_c = costs.get("fertilizer", 0); other_c = costs.get("other", 0)
        total_c = costs.get("total", seed_c + fert_c + other_c)

        st.markdown('<div class="top-plan">', unsafe_allow_html=True)
        st.markdown("### ⭐ Top Recommended Plan", unsafe_allow_html=True)
        st.markdown(f"#### 🌱 {crop_name}", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Inputs per Acre**")
            st.markdown(f"- Seed Rate: **{seed_rate}**")
            st.markdown(f"- Fertilizer (NPK): **{npk}**")
            st.markdown("---")
            st.markdown("**Cost (CAD)**")
            st.markdown(f"- Seed: **${seed_c:,.0f}**")
            st.markdown(f"- Fertilizer: **${fert_c:,.0f}**")
            if other_c:
                st.markdown(f"- Other: **${other_c:,.0f}**")
            st.markdown(f"**Total: ${total_c:,.0f}**")

        with c2:
            st.markdown("**Timeline**")
            st.markdown(f"- Planting Window: **{sched.get('planting_window','N/A')}**")
            st.markdown(f"- Harvest Window: **{sched.get('harvest_window','N/A')}**")
            if insights:
                st.markdown("---")
                st.markdown("**AI Insights**")
                st.markdown(insights)

        st.markdown('</div>', unsafe_allow_html=True)

    # --- Soil validation (pulls it above the fold & readable) ---
    if ai_validation:
        st.markdown("### 🔍 AI Soil Validation")
        st.info(ai_validation)

def create_climate_charts(climate_data: Dict[str, Any]):
    """Create visual charts for climate data"""
    if not climate_data or "daily" not in climate_data:
        return
    
    daily = climate_data["daily"]
    dates = pd.to_datetime(daily.get("time", []))
    temps = daily.get("temperature_2m_mean", [])
    precip = daily.get("precipitation_sum", [])
    
    # Aggregate by month
    df = pd.DataFrame({
        "date": dates,
        "temperature": temps,
        "precipitation": precip
    })
    df["month"] = df["date"].dt.month
    monthly = df.groupby("month").agg({
        "temperature": "mean",
        "precipitation": "sum"
    }).reset_index()
    monthly["month_name"] = monthly["month"].map({
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"
    })
    
    # Create charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_temp = px.line(monthly, x="month_name", y="temperature", 
                          title="Average Temperature by Month",
                          labels={"temperature": "Temperature (°C)", "month_name": "Month"})
        fig_temp.update_traces(line_color="#667eea", line_width=3)
        fig_temp.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white",
            font=dict(color="black", size=12),
            title_font=dict(color="black", size=14),
            xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
            yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"))
        )
        st.plotly_chart(fig_temp, use_container_width=True)
    
    with chart_col2:
        fig_precip = px.bar(monthly, x="month_name", y="precipitation",
                           title="Total Precipitation by Month",
                           labels={"precipitation": "Precipitation (mm)", "month_name": "Month"})
        fig_precip.update_traces(marker_color="#4caf50")
        fig_precip.update_layout(
            plot_bgcolor="white", 
            paper_bgcolor="white",
            font=dict(color="black", size=12),
            title_font=dict(color="black", size=14),
            xaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black")),
            yaxis=dict(title_font=dict(color="black"), tickfont=dict(color="black"))
        )
        st.plotly_chart(fig_precip, use_container_width=True)

# ================== UI ==================
st.set_page_config(page_title="Sustainable Farming", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

# Custom CSS for overall styling (includes AI overrides block at the end)
st.markdown("""
<style>

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .main-header h1 {
        color: #4caf50 !important;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
        line-height: 1.2;
        text-align: center;
    }
    .main-header p {
        display: none;
    }
    .stMetric {
        background: transparent !important;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: none;
    }
    .prediction-card {
        background: transparent !important;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: none;
        border-left: 4px solid #4caf50;
        margin-bottom: 1rem;
    }
    .location-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .image-container {
        background: transparent !important;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: none;
        margin-bottom: 1rem;
    }
    .analysis-button {
        width: 100%;
        padding: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
        border-radius: 8px;
        margin-top: 1rem;
    }
    [data-testid="stSidebar"] { display: none !important; }
    section[data-testid="stSidebar"] { display: none !important; }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
        background: transparent !important;
    }
    /* Dark background for main area */
    .main { background: #1e1e1e !important; }
    .stApp { background: #1e1e1e !important; }
    header[data-testid="stHeader"] { display: none; }

    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; border: none; border-radius: 8px;
        padding: 0.75rem 2rem; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    h3 { color: #ffffff !important; font-weight: 700; margin-top: 1.5rem; }

    .stExpander { background: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    .stExpander, .stExpander * { color: #1a1a1a !important; }
    .streamlit-expanderHeader { color: #1a1a1a !important; }

    /* Any element with white background should have black text */
    [style*="background: white"], [style*="background-color: white"],
    [style*="background:white"], [style*="background-color:white"] { color: #1a1a1a !important; }
    [style*="background: white"] *, [style*="background-color: white"] *,
    [style*="background:white"] *, [style*="background-color:white"] * { color: #1a1a1a !important; }

    /* Global dark text rules */
    h1:not(.main-header h1), h2, h3, h4, h5, h6 { color: #ffffff !important; }
    p, span, div, label { color: #ffffff !important; }
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 { color: #ffffff !important; }
    .stFileUploader label, .stFileUploader p { color: #ffffff !important; }
    [data-testid="stFileUploader"] label, [data-testid="stFileUploader"] p { color: #ffffff !important; }
    .stSelectbox label, .stSelectbox div { color: #ffffff !important; }
    [data-testid="stSelectbox"] label, [data-testid="stSelectbox"] div { color: #ffffff !important; }
    .stCaption { color: #ffffff !important; }
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: #ffffff !important; }
    .stTextInput label, [data-testid="stTextInput"] label { color: #ffffff !important; }
    .streamlit-expanderHeader { color: #ffffff !important; }
    * { color: inherit; }
    body, .main { color: #ffffff !important; }
    button, .stButton>button { color: white !important; }
    input, textarea, select { color: #1a1a1a !important; }
    [data-testid="stFileUploaderFileName"] { color: #ffffff !important; }
    [data-baseweb="select"] label, [data-baseweb="input"] label { color: #ffffff !important; }
    .main .element-container, .main .element-container * { color: #ffffff !important; }
    .stText, .stMarkdownContainer, .stMarkdownContainer * { color: #ffffff !important; }
    .uploadedFile { color: #ffffff !important; }
    input[type="text"], input[type="number"], textarea, select option { color: #1a1a1a !important; }
    label, .stTextInput label, .stSelectbox label, .stFileUploader label { color: #ffffff !important; }
    [data-testid="stFileUploader"] { background: transparent !important; }
    [data-testid="stFileUploader"] > div { background: transparent !important; }
    .stFileUploader > div { background: transparent !important; }
    [data-testid="stFileUploaderDropzone"] {
        background: transparent !important; border: 2px dashed #4caf50 !important;
    }
    [data-testid="stMetricContainer"] { background: transparent !important; }
    .element-container { background: transparent !important; }
    .block-container { background: transparent !important; }

    /* ---- AI OVERRIDES ---- */
    .ai-override * { all: revert; font-family: inherit; }
    .ai-override { color: #111; }

    .ai-override .recommendation-card{
      background:#fff; color:#111;
      border:1px solid #e9eef3; border-radius:14px;
      padding:22px; margin:16px 0 24px;
      box-shadow:0 10px 24px rgba(0,0,0,.08);
    }
    .ai-override .recommendation-card *{ color:#111 !important; }

    .ai-override .ai-section-title{
      font-size:28px; font-weight:800; letter-spacing:-.2px; color:#fff !important;
      margin:6px 0 14px 0;
    }

    .ai-override .score-bubble{
      min-width:64px; text-align:center; font-weight:800; font-size:18px;
      padding:8px 10px; border-radius:10px; background:#e8f5e9; color:#1b5e20;
      border:1px solid #c8e6c9;
    }
    .ai-override .chip{
      display:inline-flex;align-items:center;gap:6px;
      padding:6px 10px;border-radius:999px;font-size:12px;font-weight:700;
      background:#eef2ff;color:#3949ab;border:1px solid #e0e7ff;
    }

    .ai-override [data-testid="stProgress"] > div {
      background:#e9eef3 !important; border-radius:999px; height:10px;
    }
    .ai-override [data-testid="stProgress"] div[role="progressbar"]{
      background:linear-gradient(90deg,#1e88e5,#42a5f5) !important; border-radius:999px;
    }

    .ai-override .metric-row{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0 2px}
    .ai-override .metric-badge{
      display:inline-flex;align-items:center;gap:6px;
      padding:6px 10px;border-radius:999px;font-size:12px;font-weight:600;
      background:#f4f6f8;border:1px solid #e5eaef;color:#222;
    }
    .ai-override .metric-badge.water-low  { background:#e8f5e9;border-color:#c8e6c9;color:#1b5e20; }
    .ai-override .metric-badge.water-med  { background:#fff7e6;border-color:#ffe0b2;color:#e65100; }
    .ai-override .metric-badge.water-high { background:#e3f2fd;border-color:#bbdefb;color:#0d47a1; }

    .ai-override .pill{
      display:inline-block;margin:6px 6px 0 0;padding:6px 12px;border-radius:999px;
      font-size:12px;font-weight:700;border:1px solid #e0e3e7;background:#f7f9fb;color:#222;
    }
    .ai-override .pill--action { background:#4caf50; border-color:#43a047; color:#fff; }

    .ai-override .cost-wrap{margin-top:8px;border-top:1px dashed #e5eaef;padding-top:8px}
    .ai-override .cost-line{display:flex;justify-content:space-between;margin:4px 0;font-size:14px}
    .ai-override .cost-total{font-weight:800;font-size:16px;margin-top:6px}
    /* Force RED text inside recommendation cards */
    .ai-override .recommendation-card,
    .ai-override .recommendation-card * {
    color:#d32f2f !important;  /* rich red */
    }

    /* Extra specificity for headings */
    .ai-override .recommendation-card h1,
    .ai-override .recommendation-card h2,
    .ai-override .recommendation-card h3,
    .ai-override .recommendation-card h4,
    .ai-override .recommendation-card h5,
    .ai-override .recommendation-card h6 {
    color:#d32f2f !important;
    }
    .ai-override .top-plan{
      background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
      border-radius:14px;padding:22px;margin:18px 0 4px;
      color:#fff;border:1px solid rgba(255,255,255,.2);
      box-shadow:0 10px 24px rgba(0,0,0,.10);
    }
    .ai-override .top-plan *{ color:#fff !important; }

    .ai-override .constrained{max-width:980px;margin:0 auto;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header"><h1>🌱 Sustainable Farming</h1></div>', unsafe_allow_html=True)

# Default values for kNN model settings
data_dir = "data"
accept = 0.78
show_top1_on_lowconf = True

# Location & Climate section in main area (collapsible)
with st.expander("📍 Location & Climate Settings", expanded=False):
    st.markdown("Enter your farm location to get personalized crop recommendations based on local climate data.")
    loc_col1, loc_col2 = st.columns(2)
    with loc_col1:
        lat_in = st.text_input("Farm Latitude", value="44.6488", help="Enter the latitude of your farm location", key="lat_input")
    with loc_col2:
        lon_in = st.text_input("Farm Longitude", value="-63.5752", help="Enter the longitude of your farm location", key="lon_input")

# Load / build index (automatic)
try:
    with st.spinner("Loading index..."):
        X, meta = ensure_index(data_dir)
except Exception as e:
    st.error(f"Index error: {e}")
    st.stop()

labels = meta["labels"]
paths = meta["paths"]
classes = meta["classes"]

# Upload image section
st.markdown("### 📸 Upload Soil Image")
upload_col1, upload_col2 = st.columns([2, 1])
with upload_col1:
    uploaded = st.file_uploader("Upload a soil image (JPG/PNG)", type=["jpg","jpeg","png"], label_visibility="collapsed")
with upload_col2:
    sample_choice = st.selectbox("Or pick a sample", ["(none)"] + paths, label_visibility="visible")

img_to_use = None
img_origin_path = None

if uploaded is not None:
    img_to_use = Image.open(uploaded)
    img_origin_path = "(uploaded)"
elif sample_choice != "(none)":
    try:
        img_to_use = Image.open(sample_choice)
        img_origin_path = sample_choice
    except Exception as e:
        st.error(f"Failed to open selected sample: {e}")

if img_to_use is not None:
    # Main content layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(img_to_use, caption="Selected image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Complete Analysis button under the image
        button_clicked = st.button("🔬 Complete Analysis", type="primary", use_container_width=True, key="complete_analysis")
        
        if button_clicked:
            st.session_state.run_analysis = True
        
        if 'run_analysis' not in st.session_state:
            st.session_state.run_analysis = False

    # Predict
    idxs, sims = knn_predict(img_to_use, X, labels, k=3)
    top_idx = idxs[0]
    top_label = labels[top_idx]
    top_sim = float(sims[0])
    low_conf = top_sim < accept

    label_to_show = "unknown" if (low_conf and not show_top1_on_lowconf) else top_label

    src = uploaded.name if uploaded is not None else (img_origin_path or "(unknown)")
    log_prediction("predictions_log.csv", src, label_to_show, top_label, top_sim, low_conf)

    with col2:
        st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
        st.subheader("🔍 kNN Soil Prediction")
        note = " (low confidence)" if (low_conf and label_to_show != "unknown") else ""
        
        pred_col1, pred_col2 = st.columns(2)
        with pred_col1:
            st.metric("Label", f"{label_to_show}{note}")
        with pred_col2:
            st.metric("Similarity", f"{top_sim:.3f}")
        
        th_caption = f"Threshold: {accept:.2f} • {'showing top-1 only' if (low_conf and show_top1_on_lowconf) else 'unknown below threshold'}"
        st.caption(th_caption)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("### Top-3 Nearest Examples")
        ncols = st.columns(3)
        for i, (j, s) in enumerate(zip(idxs, sims)):
            try:
                with ncols[i]:
                    st.image(Image.open(paths[j]), caption=f"{labels[j]} · sim {float(s):.3f}", use_container_width=True)
            except Exception:
                pass

        # --- AI Agronomy Engine ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.divider()
    
    # Run analysis when button is clicked (moved outside columns for centering)
    if st.session_state.get('run_analysis', False):
        if label_to_show == "unknown":
            st.error("Cannot generate AI plan: kNN confidence is too low. Please add this image to memory first.")
            st.session_state.run_analysis = False
        else:
            with st.spinner(f"Fetching climate data for ({lat_in}, {lon_in})..."):
                climate_data = get_climate_data(lat_in, lon_in)

            if climate_data:
                st.success("✅ Climate data fetched!")
                
                # Display climate charts
                create_climate_charts(climate_data)

                try:
                    model = get_gemini_model()
                except Exception as e:
                    st.error(f"Error initializing Gemini: {e}")
                    st.session_state.run_analysis = False
                    st.stop()

                climate_str = json.dumps(climate_data)

                # Enhanced prompt with more fields for better UI
                master_prompt = f"""
Return ONLY a single JSON object that matches the schema below.
Rules: no prose, no Markdown fences, no comments, no trailing commas,
minified (single line), and VALID JSON. Do not omit the outermost object.

You are an expert agronomist for Eastern Canada (Nova Scotia).
Use the inputs to produce comprehensive crop recommendations.

INPUTS:
1. Soil Type (from kNN model): "{top_label}"
2. Location: Latitude {lat_in}, Longitude {lon_in}
3. Climate Data (JSON): {climate_str}

SCHEMA:
{{
  "ai_soil_validation": "Brief validation of soil type suitability (2-3 sentences)",
  "ai_recommendations": [
    {{
      "crop": "Crop name (e.g., Forage Mix, Oats, Wheat)",
      "reason": "Why this crop is suitable (1-2 sentences)",
      "score": 85,
      "yield": "Expected yield (e.g., 9.5 t/ha, 3.8 t/ha)",
      "water_requirement": "Low/Medium/High",
      "npk": "Fertilizer NPK ratio (e.g., 50-50-100, 60-30-20)",
      "seed_rate": "Seed rate per acre (e.g., 18-24 lb/ac, 80-100 lb/ac)",
      "costs": {{
        "seed": 150,
        "fertilizer": 325,
        "other": 0,
        "total": 475
      }},
      "timeline": {{
        "planting_window": "Apr-May",
        "harvest_window": "Jun, Aug, Sep"
      }}
    }}
  ],
  "top_plan": {{
    "crop_name": "Full crop name (e.g., Forage Mix (Timothy/Alfalfa/Clover))",
    "seed_rate_per_acre": "Detailed seed rate",
    "fertilizer_npk_per_acre": "NPK ratio",
    "est_cost_cad_per_acre": {{
      "seed": 150,
      "fertilizer": 325,
      "total": 475
    }},
    "schedule": {{
      "planting_window": "Apr-May",
      "harvest_window": "Jun-Aug"
    }},
    "ai_insights": "Detailed insights about why this is the top recommendation (2-3 sentences)"
  }}
}}

Provide at least 2-3 crop recommendations in ai_recommendations array.
"""

                with st.spinner("Generating analysis..."):
                    try:
                        response = model.generate_content(
                            master_prompt,
                            generation_config={
                                "response_mime_type": "application/json",
                                "temperature": 0
                            }
                        )
                        parsed_json = parse_llm_json_output(getattr(response, "text", ""))

                        if parsed_json:
                            st.success("✨ AI Plan Generated!")
                            st.markdown("<br>", unsafe_allow_html=True)

                            # ===== AI OUTPUT WRAPPER (OVERRIDES SCOPE) =====
                            st.markdown('<div class="ai-override"><div class="constrained">', unsafe_allow_html=True)
                            st.markdown('<div class="ai-section-title">🤖 AI-Powered Agronomy Plan</div>', unsafe_allow_html=True)

                            # Render beautiful cards instead of raw JSON
                            render_recommendation_cards(parsed_json)

                            st.markdown('</div></div>', unsafe_allow_html=True)
                            # =================================================

                            # Optional: Show raw JSON in expander for debugging
                            with st.expander("🔧 View Raw JSON (Debug)"):
                                st.json(parsed_json)
                            
                            # Reset button to run analysis again
                            if st.button("🔄 Run Analysis Again", key="reset_analysis"):
                                st.session_state.run_analysis = False
                                st.rerun()
                        else:
                            st.error("Failed to parse AI JSON. See raw output below.")
                            st.text_area("Raw AI Output", getattr(response, "text", ""), height=240)
                            if st.button("🔄 Try Again", key="retry_analysis"):
                                st.session_state.run_analysis = False
                                st.rerun()

                    except Exception as e:
                        st.error(f"An error occurred while calling the Gemini API: {e}")
                        try:
                            error_output = getattr(response, "text", "No response")
                        except:
                            error_output = "No response available"
                        st.text_area("Raw AI Output (on error)", error_output, height=240)
                        if st.button("🔄 Try Again", key="retry_analysis_error"):
                            st.session_state.run_analysis = False
                            st.rerun()

else:
    st.info("Upload an image or pick a sample from your data folder to run a prediction.")
