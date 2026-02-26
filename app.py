import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
import time
from streamlit_lottie import st_lottie

# -------------------- App Config --------------------
st.set_page_config(page_title="PillScan Ultra", layout="wide", initial_sidebar_state="collapsed")

# -------------------- Advanced CSS: Neon & Glass --------------------
st.markdown("""
<style>
    /* 1. Animated Gradient Background */
    .stApp {
        background: linear-gradient(-45deg, #0f2027, #203a43, #2c5364, #1a2a6c);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
    }
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* 2. Neon Glassmorphism Cards */
    .neon-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border-radius: 20px;
        padding: 30px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 0 20px rgba(0, 255, 136, 0.1);
        text-align: center;
        margin-bottom: 25px;
        transition: 0.4s;
    }
    .neon-card:hover {
        border: 1px solid #00ff88;
        box-shadow: 0 0 30px rgba(0, 255, 136, 0.4);
        transform: scale(1.02);
    }

    /* 3. Glowing Typography */
    .ultra-title {
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(to right, #00ff88, #00d4ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        filter: drop-shadow(0 0 10px rgba(0, 255, 136, 0.5));
        letter-spacing: -2px;
        text-align: center;
        margin-bottom: 0px;
    }

    /* 4. Hide Default Streamlit Junk */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# -------------------- Load Model & Animations --------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

model = load_yolo()

def load_lottie(url):
    try:
        return requests.get(url, timeout=5).json()
    except: return None

# Using high-energy colorful medical lotties
lottie_main = load_lottie("https://lottie.host/8254c0e6-990a-422e-a510-7607771746c8/E6yH6GOfG8.json")
lottie_scanning = load_lottie("https://lottie.host/f41e54c7-14e4-4112-8700-165c71a39643/9S7yO0n6yO.json")

# -------------------- Detection Logic --------------------
def get_counts(results):
    tablets = capsules = 0
    for r in results[0].boxes:
        cls = int(r.cls[0])
        if cls == 0: capsules += 1
        elif cls == 1: tablets += 1
    return tablets, capsules, tablets + capsules

# -------------------- UI Header --------------------
st.markdown('<h1 class="ultra-title">PILLSCAN ULTRA</h1>', unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#00ff88; font-weight:bold; letter-spacing:2px;'>VISION INTELLIGENCE 2026</p>", unsafe_allow_html=True)

# -------------------- Layout --------------------
col_ctrl, col_disp = st.columns([1, 2])

with col_ctrl:
    st.markdown('<div class="neon-card">', unsafe_allow_html=True)
    if lottie_main: st_lottie(lottie_main, height=150)
    mode = st.radio("INTERFACE MODE", ["📁 UPLOAD", "📸 CAMERA"])
    
    if mode == "📁 UPLOAD":
        source = st.file_uploader("", type=["jpg", "png", "jpeg"])
    else:
        source = st.camera_input("")
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------- Processing --------------------
if source:
    img = Image.open(source).convert("RGB")
    img_np = np.array(img)

    with st.spinner("⚡ NEURAL OVERDRIVE ACTIVATED..."):
        results = model(img_np)
        annotated = results[0].plot()
        tabs, caps, total = get_counts(results)
    
    with col_disp:
        # Image Display with neon border
        st.markdown('<div class="neon-card">', unsafe_allow_html=True)
        st.image(annotated, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Colorful Stats Row
        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.markdown(f'<div class="neon-card"><h4 style="color:#9ea4af;">TOTAL</h4><h1 style="color:#00ff88;">{total}</h1></div>', unsafe_allow_html=True)
        with stat2:
            st.markdown(f'<div class="neon-card"><h4 style="color:#9ea4af;">CAPSULES</h4><h1 style="color:#00d4ff;">{caps}</h1></div>', unsafe_allow_html=True)
        with stat3:
            st.markdown(f'<div class="neon-card"><h4 style="color:#9ea4af;">TABLETS</h4><h1 style="color:#ff00ff;">{tabs}</h1></div>', unsafe_allow_html=True)
        
        st.balloons()
else:
    with col_disp:
        if lottie_scanning:
            st_lottie(lottie_scanning, height=500)
        else:
            st.info("System Ready. Awaiting Visual Input...")
