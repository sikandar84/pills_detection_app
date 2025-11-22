import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Pill Counter App", layout="wide")

# Load Model
model = YOLO("best.pt")

# Custom CSS for Modern UI
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


# ---------------------------------------------------------
# HELPER FUNCTION: Correct Class Mapping
# ---------------------------------------------------------
def get_counts(results):
    tablets = capsules = 0
    for r in results[0].boxes:
        cls = int(r.cls[0])
        if cls == 1:      # Tablet
            tablets += 1
        elif cls == 0:    # Capsule
            capsules += 1
    return tablets, capsules, tablets + capsules


# ---------------------------------------------------------
# IMAGE UPLOAD SECTION (FIRST)
# ---------------------------------------------------------
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



# ---------------------------------------------------------
# LIVE CAMERA SECTION (BELOW)
# ---------------------------------------------------------
st.subheader("🎥 Real-time Detection (Live Camera)")

camera = st.checkbox("Start Live Camera")

if camera:
    frame_window = st.empty()
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Camera not detected! Please check your webcam.")
    else:
        while camera:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to access camera!")
                break

            # Run YOLO
            results = model(frame)

            tablets, capsules, total = get_counts(results)

            annotated = results[0].plot()

            # Add text overlay
            cv2.putText(
                annotated,
                f"Tablets: {tablets} | Capsules: {capsules} | Total: {total}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2
            )

            # Display live video
            frame_window.image(annotated, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()
