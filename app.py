import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
import time
from streamlit_lottie import st_lottie
# -------------------- App Config --------------------
st.set_page_config(page_title="PillScan Pro", layout="wide", initial_sidebar_state="collapsed")

# -------------------- Advanced Bio-Tech CSS --------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@200;800&family=Space+Grotesk:wght@300;700&display=swap');

    /* 1. Prevent scrolling & set base theme */
    .block-container { padding: 1rem 2rem !important; height: 100vh; }
    .stApp {
        background: radial-gradient(circle at 50% 50%, #0a192f 0%, #020617 100%);
        overflow: hidden;
    }

    /* 2. Dynamic "Breathing" Title */
    .dynamic-title {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0px;
        background: linear-gradient(90deg, #00f2fe, #4facfe, #00f2fe);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 3s linear infinite, breathe 4s ease-in-out infinite;
    }

    @keyframes shine { to { background-position: 200% center; } }
    @keyframes breathe {
        0%, 100% { transform: scale(1); opacity: 0.9; }
        50% { transform: scale(1.02); opacity: 1; filter: drop-shadow(0 0 15px rgba(79, 172, 254, 0.6)); }
    }

    /* 3. Modernized Neon Cards */
    .health-card {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 15px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s ease;
    }
    .health-card:hover {
        border: 1px solid rgba(0, 242, 254, 0.5);
        background: rgba(255, 255, 255, 0.04);
    }

    /* 4. Scanning Image Constraint */
    .stImage > img {
        max-height: 48vh !important;
        object-fit: contain;
        border-radius: 12px;
        border: 1px solid rgba(0, 242, 254, 0.3);
        box-shadow: 0 0 20px rgba(0, 242, 254, 0.2);
    }

    /* 5. Clean Sidebar */
    [data-testid="stSidebar"] { background-color: #020617; border-right: 1px solid #1e293b; }
</style>
""", unsafe_allow_html=True)

# -------------------- Logic & Models --------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

model = load_yolo()

def load_lottie(url):
    try: return requests.get(url, timeout=5).json()
    except: return None

lottie_hero = load_lottie("https://lottie.host/80860541-118f-495f-9e8c-84381e4b868e/vV6kOonhGk.json")

def get_counts(results):
    tablets = capsules = 0
    for r in results[0].boxes:
        cls = int(r.cls[0])
        if cls == 0: capsules += 1
        elif cls == 1: tablets += 1
    return tablets, capsules, tablets + capsules

# -------------------- UI Header --------------------
st.markdown('<h1 class="dynamic-title">PillScan Ultra</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#94a3b8; font-family:Space Grotesk; margin-top:-10px; letter-spacing:3px; font-size:0.8rem;'>PRECISION PHARMA ANALYTICS v2.6</p>", unsafe_allow_html=True)

# -------------------- Dashboard Layout --------------------
col_input, col_view = st.columns([1, 2.2], gap="medium")

with col_input:
    st.markdown('<div class="health-card">', unsafe_allow_html=True)
    if lottie_hero:
        st_lottie(lottie_hero, height=120, key="hero")
    
    mode = st.radio("ANALYSIS SOURCE", ["📁 UPLOAD", "📸 CAMERA"], horizontal=True)
    if mode == "📁 UPLOAD":
        source = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        source = st.camera_input("")
    
    st.markdown('<hr style="opacity:0.1; margin:15px 0;">', unsafe_allow_html=True)
    st.markdown("<p style='color:#64748b; font-size:0.75rem;'>AI Model: YOLO-Pill v8.1<br>Status: Ready</p>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Results Section --------------------
if source:
    img = Image.open(source).convert("RGB")
    
    with st.status("🧠 Processing Neural Layers...", expanded=False) as status:
        results = model(np.array(img))
        annotated = results[0].plot()
        tabs, caps, total = get_counts(results)
        time.sleep(0.4)
        status.update(label="Analysis Verified", state="complete")

    with col_view:
        # Top Stats Row
        s1, s2, s3 = st.columns(3)
        s1.markdown(f'<div class="health-card"><small style="color:#94a3b8">TOTAL UNITS</small><h2 style="color:#f8fafc; margin:0;">{total}</h2></div>', unsafe_allow_html=True)
        s2.markdown(f'<div class="health-card"><small style="color:#94a3b8">CAPSULES</small><h2 style="color:#38bdf8; margin:0;">{caps}</h2></div>', unsafe_allow_html=True)
        s3.markdown(f'<div class="health-card"><small style="color:#94a3b8">TABLETS</small><h2 style="color:#fbbf24; margin:0;">{tabs}</h2></div>', unsafe_allow_html=True)
        
        # Main Visual
        st.image(annotated, use_column_width=True)
        
        # Bottom Utility Bar
        u1, u2 = st.columns([2, 1])
        with u1:
            confidence = np.mean(results[0].boxes.conf.cpu().numpy()) if total > 0 else 0
            st.progress(float(confidence), text=f"Detection Confidence: {confidence:.1%}")
        with u2:
            if st.button("🔄 Reset System", use_container_width=True):
                st.rerun()
else:
    with col_view:
        st.markdown("""
        <div style="height: 55vh; display: flex; flex-direction: column; align-items: center; justify-content: center; background: rgba(255,255,255,0.01); border: 1px dashed rgba(255,255,255,0.1); border-radius: 20px;">
            <div style="font-size: 3rem; margin-bottom: 10px; opacity: 0.5;">📡</div>
            <p style="color: #475569; font-family: Space Grotesk; letter-spacing: 1px;">AWAITING OPTICAL FEED...</p>
        </div>
        """, unsafe_allow_html=True)
