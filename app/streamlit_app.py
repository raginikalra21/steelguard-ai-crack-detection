import sys
import os
import tempfile

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from src.explainability import get_gradcam, overlay_heatmap

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="SteelGuard AI", layout="wide")

st.title("🏭 SteelGuard AI – Crack Detection System")
st.write("Upload a steel surface image for crack inspection")

# ---------------- LOAD MODEL ----------------
model = load_model("models/best_resnet50_crack_detector.h5")

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- PREDICTION BLOCK ----------------
if uploaded_file is not None:
    # Read uploaded image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Preprocess image
    img_resized = cv2.resize(img, (224, 224))
    img_input = img_resized.astype("float32") / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Get prediction score
    pred = float(model.predict(img_input, verbose=0)[0][0])

    # Safety-biased threshold
    threshold = 0.6

    if pred < threshold:
        label = "⚠️ Crack Detected"
        confidence = (1 - pred) * 100
    else:
        label = "✅ No Crack"
        confidence = pred * 100

    # Show prediction
    st.subheader(f"Prediction: {label}")
    st.write(f"Confidence: {confidence:.2f}%")
    st.write(f"Raw Score: {pred:.4f}")

    # Generate Grad-CAM
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        cv2.imwrite(tmp.name, img_resized)
        heatmap = get_gradcam(model, img_input)
        result = overlay_heatmap(tmp.name, heatmap)

    # Display images side by side
    col1, col2 = st.columns(2)

    with col1:
        st.image(
            img_resized,
            caption="Original Image",
            use_container_width=True
        )

    with col2:
        st.image(
            result,
            caption="Grad-CAM Heatmap",
            use_container_width=True
        )