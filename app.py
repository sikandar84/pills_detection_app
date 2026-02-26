import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# -------------------- App Config --------------------
st.set_page_config(page_title="PillScan Pro", layout="wide", initial_sidebar_state="collapsed")

# -------------------- Ultra-Compact CSS --------------------
st.markdown("""
<style>
    /* 1. Fix app height to prevent scrolling */
    .block-container {
        padding-top: 1rem !important;
        padding-bottom: 0rem !important;
        height: 100vh;
    }
    
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364);
        overflow: hidden; /* Disable scroll */
    }

    /* 2. Compact Neon Cards */
    .neon-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(15px);
        border-radius: 12px;
        padding: 10px 15px;
        border: 1px solid rgba(0, 255, 136, 0.2);
        text-align: center;
        margin-bottom: 10px;
    }

    /* 3. Constrain Image Height */
    .stImage > img {
        max-height: 45vh !important;
        width: auto !important;
        margin-left: auto;
        margin-right: auto;
        border-radius: 10px;
        border: 2px solid #00ff88;
    }

    .ultra-title {
        font-size: 2.2rem !important;
        font-weight: 800;
        background: linear-gradient(to right, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: -20px;
    }
    
    /* Minimize spacing */
    .stMarkdown div { line-height: 1.2; }
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model & Animations --------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

model = load_yolo()

def load_lottie(url):
    try: return requests.get(url, timeout=5).json()
    except: return None

lottie_health = load_lottie("https://lottie.host/8254c0e6-990a-422e-a510-7607771746c8/E6yH6GOfG8.json")

# -------------------- Detection Logic --------------------
def get_counts(results):
    tablets = capsules = 0
    for r in results[0].boxes:
        cls = int(r.cls[0])
        if cls == 0: capsules += 1
        elif cls == 1: tablets += 1
    return tablets, capsules, tablets + capsules

# -------------------- Header --------------------
st.markdown('<h1 class="ultra-title">PILLSCAN PRO</h1>', unsafe_allow_html=True)

# -------------------- Main Interface --------------------
# Split screen into Control (Left) and Results (Right)
col_left, col_right = st.columns([1, 2.5])

with col_left:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    if lottie_health:
        st_lottie(lottie_health, height=100, key="top_ani")
    
    mode = st.radio("SELECT INPUT", ["📁 UPLOAD", "📸 CAMERA"], horizontal=True)
    
    if mode == "📁 UPLOAD":
        source = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        source = st.camera_input("")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Processing & Output --------------------
if source:
    img = Image.open(source).convert("RGB")
    results = model(np.array(img))
    annotated = results[0].plot()
    tabs, caps, total = get_counts(results)
    
    with col_right:
        # 1. Metrics Row (Top)
        m1, m2, m3 = st.columns(3)
        m1.markdown(f'<div class="neon-card"><small>TOTAL</small><h2 style="color:#00ff88; margin:0;">{total}</h2></div>', unsafe_allow_html=True)
        m2.markdown(f'<div class="neon-card"><small>CAPSULES</small><h2 style="color:#00d4ff; margin:0;">{caps}</h2></div>', unsafe_allow_html=True)
        m3.markdown(f'<div class="neon-card"><small>TABLETS</small><h2 style="color:#ff00ff; margin:0;">{tabs}</h2></div>', unsafe_allow_html=True)
        
        # 2. Scanned Image (Middle) - CSS Constrained to 45% of Screen Height
        st.image(annotated, use_column_width=False)
        
        # 3. Quick Action
        if st.button("🗑️ Clear Scan", use_container_width=True):
            st.rerun()
else:
    with col_right:
        st.markdown("""
        <div style="height: 60vh; display: flex; align-items: center; justify-content: center; border: 2px dashed rgba(255,255,255,0.1); border-radius: 20px;">
            <p style="color: #666; font-size: 1.2rem;">Awaiting Image Input for Neural Analysis...</p>
        </div>
        """, unsafe_allow_html=True)
