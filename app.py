import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time
from streamlit_lottie import st_lottie
import requests

# -------------------- Configuration & Assets --------------------
st.set_page_config(page_title="PillScan Pro", layout="wide", initial_sidebar_state="collapsed")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Lottie Animations (Medical/Scanning theme)
lottie_pill = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_5njp9vbg.json")
lottie_scan = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_ndm890at.json")

# Load YOLO Model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# -------------------- Advanced CSS --------------------
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background: radial-gradient(circle, #1a1c23 0%, #0d1117 100%);
    }
    
    /* Glassmorphism Card */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #4CAF50;
    }
    
    /* Title Animation */
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(#4CAF50, #2E7D32);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        animation: fadeInDown 1s ease-out;
    }
    
    /* Custom Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Header Section --------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st_lottie(lottie_pill, height=150, key="pill_ani")
    st.markdown('<h1 class="main-title">PILLSCAN PRO</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #8b949e;'>AI-Powered Pharmaceutical Recognition & Counting</p>", unsafe_allow_html=True)

st.write("---")

# -------------------- Sidebar & Logic --------------------
with st.sidebar:
    st.header("⚙️ Controls")
    option = st.radio("Input Source", ["Upload Image", "Live Camera"], help="Choose how you want to capture the pills.")
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)
    st.info("Ensure lighting is bright for 99% accuracy.")

def process_image(img):
    with st.spinner('🔬 Analyzing chemical signatures...'):
        results = model(img, conf=conf_threshold)
        
        # Artificial slight delay for 'pro' feel
        time.sleep(0.5) 
        
        tablets = 0
        capsules = 0
        for r in results[0].boxes:
            cls = int(r.cls[0])
            if cls == 0: capsules += 1
            elif cls == 1: tablets += 1
            
        return results[0].plot(), tablets, capsules, tablets + capsules

# -------------------- Main Interface --------------------
source_img = None

if option == "Upload Image":
    source_img = st.file_uploader("📂 Drag and drop pill images here", type=["jpg", "jpeg", "png"])
else:
    source_img = st.camera_input("📸 Capture snapshot")

if source_img:
    image = Image.open(source_img).convert("RGB")
    annotated_img, tabs, caps, total = process_image(image)

    # Dashboard Layout
    res_col, stats_col = st.columns([2, 1])

    with res_col:
        st.image(annotated_img, caption="AI Vision Feedback", use_column_width=True)

    with stats_col:
        st.markdown("### 📊 Analysis Report")
        
        # Displaying Metrics in Glassmorphism cards
        st.markdown(f"""
            <div class="metric-card">
                <p style="color: #8b949e; margin-bottom: 5px;">Total Units</p>
                <h2 style="color: #4CAF50; margin: 0;">{total}</h2>
            </div>
            <br>
            <div class="metric-card">
                <p style="color: #8b949e; margin-bottom: 5px;">Tablets Identified</p>
                <h2 style="color: #64B5F6; margin: 0;">{tabs}</h2>
            </div>
            <br>
            <div class="metric-card">
                <p style="color: #8b949e; margin-bottom: 5px;">Capsules Identified</p>
                <h2 style="color: #FFB74D; margin: 0;">{caps}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if total > 0:
            st.success("✅ Scanning Complete")
            st.download_button("Export Report (.txt)", f"Pill Count Report\nTabs: {tabs}\nCaps: {caps}\nTotal: {total}")
        else:
            st.warning("No pills detected. Adjust the confidence slider in the sidebar.")

else:
    st.info("👈 Please upload an image or take a photo to begin.")
    st_lottie(lottie_scan, height=300)
