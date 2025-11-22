import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Pill Counter App", layout="wide")

# Load Model
model = YOLO("best.pt")

# Custom CSS
st.markdown("""
<style>
    .main {background: #0d1117; color: white;}
    .title {text-align: center; font-size: 38px; font-weight: bold; color: #4CAF50;}
    .counter-box {
        padding: 15px;
        background: #1e1e1e;
        border-radius: 12px;
        margin-top: 10px;
        font-size: 20px;
        color: white;
        text-align: center;
        border: 2px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">💊 Smart Pill & Capsule Detection</p>', unsafe_allow_html=True)

st.info("📌 Note: Real-time camera mode is disabled on Streamlit Cloud since webcam access is not supported.")

# Helper function
def get_counts(results):
    pills = capsules = 0
    for r in results[0].boxes:
        cls = int(r.cls[0])
        if cls == 0:
            pills += 1
        elif cls == 1:
            capsules += 1
    return pills, capsules, pills + capsules

# IMAGE MODE ONLY (Works everywhere)
st.subheader("📸 Upload a Pill / Capsule Image")
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    results = model(img_np)
    annotated = results[0].plot()

    pills, capsules, total = get_counts(results)

    st.image(annotated, caption="Detection Results", use_column_width=True)

    st.markdown(f"""
    <div class="counter-box">
        Tablets: {pills} <br>
        Capsules: {capsules} <br>
        <b>Total: {total}</b>
    </div>
    """, unsafe_allow_html=True)
