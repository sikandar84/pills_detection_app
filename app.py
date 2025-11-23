import streamlit as st
import numpy as np
from ultralytics import YOLO
from PIL import Image

# -------------------- Streamlit Page Setup --------------------
st.set_page_config(page_title="Pill Counter App", layout="wide")

# Load YOLO Model
model = YOLO("best.pt")  # Make sure best.pt is in the same repo

# -------------------- Custom CSS --------------------
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
.upload-box {
    border: 2px dashed #4CAF50;
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="title">💊 Smart Pill & Capsule Detection</p>', unsafe_allow_html=True)

# -------------------- Mode Selection --------------------
option = st.sidebar.selectbox("Select Mode", ["Image Upload", "Camera Snapshot"])

# -------------------- Helper Function --------------------
# YOLO CLASS 0 = CAPSULE, CLASS 1 = TABLET
def get_counts(results):
    tablets = capsules = 0
    for r in results[0].boxes:
        cls = int(r.cls[0])
        if cls == 0:
            capsules += 1
        elif cls == 1:
            tablets += 1
    return tablets, capsules, tablets + capsules

# -------------------- IMAGE UPLOAD MODE --------------------
if option == "Image Upload":
    st.subheader("📸 Upload an Image")
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        img_np = np.array(image)

        results = model(img_np)
        annotated = results[0].plot()

        tablets, capsules, total = get_counts(results)

        st.image(annotated, caption="Detection Results", use_column_width=True)

        st.markdown(f"""
        <div class="counter-box">
        Tablets: {tablets} <br>
        Capsules: {capsules} <br>
        <b>Total: {total}</b>
        </div>
        """, unsafe_allow_html=True)

# -------------------- CAMERA SNAPSHOT MODE --------------------
else:
    st.subheader("📷 Camera Snapshot (Mobile Friendly)")
    img = st.camera_input("Open Camera")

    if img:
        image = Image.open(img).convert("RGB")
        img_np = np.array(image)

        results = model(img_np)
        annotated = results[0].plot()

        tablets, capsules, total = get_counts(results)

        st.image(
            annotated,
            caption=f"Tablets: {tablets} | Capsules: {capsules} | Total: {total}",
            use_column_width=True
        )
        st.markdown(f"""
        <div class="counter-box">
        Tablets: {tablets} <br>
        Capsules: {capsules} <br>
        <b>Total: {total}</b>
        </div>
        """, unsafe_allow_html=True)
