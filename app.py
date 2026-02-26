import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image
import requests
from streamlit_lottie import st_lottie

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="PillScan Pro", layout="wide", initial_sidebar_state="collapsed")

# -------------------- Load Assets & Model --------------------
@st.cache_resource
def load_yolo():
    return YOLO("best.pt")

model = load_yolo()

def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except:
        return None

lottie_health = load_lottieurl("https://lottie.host/80860541-118f-495f-9e8c-84381e4b868e/vV6kOonhGk.json")
lottie_scan = load_lottieurl("https://lottie.host/936a1662-d499-4d6a-861f-172c39e9487c/kP8U6v9F6u.json")

# -------------------- Custom CSS (Modern Health Style) --------------------
st.markdown("""
<style>
    /* Dark Slate Background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Glassmorphism Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #4CAF50;
    }
    
    .title-text {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        background: linear-gradient(90deg, #4CAF50, #81C784);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Header Section --------------------
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if lottie_health:
        st_lottie(lottie_health, height=150, key="pill_ani")
    st.markdown('<h1 class="title-text">PILLSCAN PRO</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #9ea4af;'>Smart Pharmaceutical Recognition System</p>", unsafe_allow_html=True)

st.write("---")

# -------------------- Your Original Logic --------------------
def get_counts(results):
    tablets = capsules = 0
    for r in results[0].boxes:
        cls = int(r.cls[0])
        if cls == 0:
            capsules += 1
        elif cls == 1:
            tablets += 1
    return tablets, capsules, tablets + capsules

# -------------------- Mode Selection --------------------
option = st.sidebar.selectbox("Select Input Mode", ["Image Upload", "Camera Snapshot"])
st.sidebar.divider()
st.sidebar.info("Accuracy tip: Use a high-contrast background (like a dark tray).")

# -------------------- UI Logic --------------------
source_file = None

if option == "Image Upload":
    st.subheader("📸 Upload Scan")
    source_file = st.file_uploader("Choose a pill image", type=["jpg", "png", "jpeg"])
else:
    st.subheader("📷 Camera Snapshot")
    source_file = st.camera_input("Focus on the pills")

# -------------------- Main Execution --------------------
if source_file:
    image = Image.open(source_file).convert("RGB")
    img_np = np.array(image)

    # Performance Spinner
    with st.spinner('🔬 Running AI Diagnostics...'):
        results = model(img_np)
        annotated = results[0].plot()
        tablets, capsules, total = get_counts(results)

    # Dashboard Layout
    res_col, stats_col = st.columns([2, 1])

    with res_col:
        st.image(annotated, caption="Computer Vision Feedback", use_column_width=True)

    with stats_col:
        st.markdown("### 📊 Live Count")
        
        # Displaying Counts in Glass Cards
        st.markdown(f"""
            <div class="metric-card">
                <span style="color:#81C784; font-size: 0.9rem;">TOTAL PILLS</span>
                <h1 style="color:white; margin:0;">{total}</h1>
            </div>
            <br>
            <div class="metric-card" style="border-left: 4px solid #4CAF50;">
                <span style="color:#9ea4af;">💊 Capsules</span>
                <h2 style="color:white; margin:0;">{capsules}</h2>
            </div>
            <br>
            <div class="metric-card" style="border-left: 4px solid #64B5F6;">
                <span style="color:#9ea4af;">⚪ Tablets</span>
                <h2 style="color:white; margin:0;">{tablets}</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if total > 0:
            st.success("✅ Analysis Complete")
        else:
            st.warning("No items detected.")

else:
    # Idle state animation
    st.divider()
    if lottie_scan:
        st_lottie(lottie_scan, height=300, key="scan_idle")
