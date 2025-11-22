import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import time

st.set_page_config(page_title="Pill Counter App", layout="wide")

# Load Model
model = YOLO("best.pt")

# CSS Style
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

# Title
st.markdown('<p class="title">💊 Smart Pill & Capsule Detection</p>', unsafe_allow_html=True)

# Correct Class Mapping
# 0 = Capsule, 1 = Tablet
CLASS_NAMES = {0: "Capsule", 1: "Tablet"}

# Count Pills
def get_counts(result):
    tablets = capsules = 0
    for box in result[0].boxes:
        cls = int(box.cls[0])
        if CLASS_NAMES[cls] == "Tablet":
            tablets += 1
        else:
            capsules += 1
    return tablets, capsules, tablets + capsules


# Select Mode
option = st.sidebar.selectbox("Mode", ["📱 Phone Camera Live", "📸 Upload Image"])


# 📱 Live Mobile Camera Mode -----------------------------------------------
if option == "📱 Phone Camera Live":
    st.info("📌 Use your phone to take pictures. The app will refresh every 1 sec.")

    img_file_buffer = st.camera_input("Camera Feed")

    if img_file_buffer:
        image = Image.open(img_file_buffer).convert("RGB")
        img_np = np.array(image)

        result = model(img_np)
        tablets, capsules, total = get_counts(result)
        annotated = result[0].plot()

        st.image(annotated, channels="RGB", use_column_width=True)

        st.markdown(f"""
        <div class="counter-box">
            Tablets: {tablets} <br>
            Capsules: {capsules} <br>
            <b>Total: {total}</b>
        </div>
        """, unsafe_allow_html=True)

    # Auto refresh
    time.sleep(1)
    st.rerun()


# 📸 Upload Image Mode ------------------------------------------------------
else:
    st.subheader("Upload Pill Image")
    uploaded_file = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        result = model(img_np)
        tablets, capsules, total = get_counts(result)
        annotated = result[0].plot()

        st.image(annotated, caption="Results", use_column_width=True)

        st.markdown(f"""
        <div class="counter-box">
            Tablets: {tablets} <br>
            Capsules: {capsules} <br>
            <b>Total: {total}</b>
        </div>
        """, unsafe_allow_html=True)
