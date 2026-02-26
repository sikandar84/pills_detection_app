import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
import time
from streamlit_lottie import st_lottie

# -------------------- Configuration --------------------
st.set_page_config(page_title="PillScan Pro", layout="wide", initial_sidebar_state="expanded")

# --- Function to load Lottie animations with error handling ---
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

# Load Animations
lottie_pill = load_lottieurl("https://lottie.host/80860541-118f-495f-9e8c-84381e4b868e/vV6kOonhGk.json") # Reliable URL
lottie_scan = load_lottieurl("https://lottie.host/936a1662-d499-4d6a-861f-172c39e9487c/kP8U6v9F6u.json")

# --- Load YOLO Model (Cached to prevent reloading) ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: Make sure 'best.pt' is in the same folder as this script. Error: {e}")
    st.stop()

# -------------------- Advanced UI Styling --------------------
st.markdown("""
<style>
    /* Gradient Background */
    .stApp {
        background: radial-gradient(circle at top, #1a2a6c, #b21f1f, #fdbb2d);
        background-attachment: fixed;
    }
    
    /* Glassmorphism Effect */
    .glass-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(15px);
        -webkit-backdrop-filter: blur(15px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 20px;
        padding: 25px;
        color: white;
        text-align: center;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        margin-bottom: 20px;
    }

    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if lottie_pill:
        st_lottie(lottie_pill, height=180, key="main_ani")
    else:
        st.markdown("<h1 style='text-align: center;'>💊</h1>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: center; color: white;'>PILLSCAN PRO</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #f0f0f0;'>Advanced Pharmaceutical Computer Vision</p>", unsafe_allow_html=True)

# -------------------- Sidebar Controls --------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3028/3028573.png", width=100)
    st.title("Settings")
    mode = st.radio("Select Mode", ["📁 Upload Image", "📸 Camera Capture"])
    conf_level = st.slider("AI Sensitivity (Confidence)", 0.1, 1.0, 0.25)
    st.divider()
    st.info("Tip: Use a plain, dark background for best results.")

# -------------------- Logic & Processing --------------------
source = None
if mode == "📁 Upload Image":
    source = st.file_uploader("Upload pill photo...", type=["jpg", "png", "jpeg"])
else:
    source = st.camera_input("Take a photo of the pills")

if source:
    img = Image.open(source).convert("RGB")
    
    with st.status("🔍 Analyzing samples...", expanded=True) as status:
        # AI Inference
        results = model(img, conf=conf_level)
        annotated_img = results[0].plot()
        
        # Extract counts
        caps = sum(1 for box in results[0].boxes if int(box.cls[0]) == 0)
        tabs = sum(1 for box in results[0].boxes if int(box.cls[0]) == 1)
        total = caps + tabs
        
        time.sleep(1) # For dramatic effect
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # Display Results in Modern Layout
    col_img, col_stats = st.columns([2, 1])

    with col_img:
        st.image(annotated_img, caption="AI Detection Result", use_column_width=True)

    with col_stats:
        st.markdown("### Results Summary")
        
        # Metrics using custom Glassmorphism CSS
        st.markdown(f"""
        <div class="glass-card">
            <small>TOTAL UNITS</small>
            <h1 style="font-size: 50px; margin:0; color:#00ff88;">{total}</h1>
        </div>
        <div class="glass-card" style="border-left: 5px solid #ffcc00;">
            <p style="margin:0;">💊 Capsules: <b>{caps}</b></p>
        </div>
        <div class="glass-card" style="border-left: 5px solid #00d4ff;">
            <p style="margin:0;">⚪ Tablets: <b>{tabs}</b></p>
        </div>
        """, unsafe_allow_html=True)
        
        if total > 0:
            st.balloons()
            st.success("Verification successful.")
        else:
            st.warning("No units detected. Try lowering the sensitivity in the sidebar.")

else:
    # Empty State with Animation
    st.divider()
    if lottie_scan:
        st_lottie(lottie_scan, height=300, key="idle_ani")
    else:
        st.write("Awaiting Input...")
