import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import datetime
import base64
import os
from io import BytesIO
import tensorflow as tf   # ✅ IMPORTANT

# PAGE CONFIG
st.set_page_config(page_title="SteelGuard AI", layout="wide")

CLASS_NAMES = [
    "crazing","inclusion","patches",
    "pitted_surface","rolled-in_scale","scratches"
]

# MODEL LOADING
@st.cache_resource
def load_model_safe():
    from tensorflow.keras.models import load_model
    paths = [
        "models/resnet50_crack_detector.h5",
        "models/best_model.keras"
    ]
    for p in paths:
        if os.path.exists(p):
            return load_model(p), p
    return None, None

model, model_path = load_model_safe()

# ✅ DEFINE FIRST (FIX)
MODEL_LOADED = model is not None
DEMO_MODE = not MODEL_LOADED

# ================== GRAD CAM ==================
grad_model = None

if MODEL_LOADED:
    base_model = model.layers[0]

    x = base_model.output
    for layer in model.layers[1:]:
        x = layer(x)

    functional_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=x
    )

    last_conv = base_model.get_layer("conv5_block3_out")

    grad_model = tf.keras.models.Model(
        inputs=functional_model.input,
        outputs=[last_conv.output, functional_model.output]
    )

def get_gradcam(img_array):
    if grad_model is None:
        return None

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        idx = tf.argmax(preds[0])
        class_channel = preds[:, idx]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap)+1e-8)

    return heatmap

# ================= UI =================

st.title("🛡️ SteelGuard AI")
st.write("Defect Detection + Explainable AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    img_resized = cv2.resize(img_np,(224,224))/255.0
    img_input = np.expand_dims(img_resized, axis=0)

    if MODEL_LOADED:
        preds = model.predict(img_input)[0]
    else:
        preds = np.random.dirichlet(np.ones(6))

    pred_idx = np.argmax(preds)
    label = CLASS_NAMES[pred_idx]
    conf = preds[pred_idx]

    # GradCAM
    heatmap = get_gradcam(img_input)

    col1,col2 = st.columns(2)

    with col1:
        st.image(image, caption="Original")

    with col2:
        if heatmap is not None:
            heatmap = cv2.resize(heatmap,(224,224))
            heatmap = np.uint8(255*heatmap)
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(
                cv2.resize(img_np,(224,224)),0.6,
                heatmap,0.4,0
            )

            st.image(overlay, caption="Grad-CAM")
        else:
            st.info("GradCAM unavailable")

    st.success(f"{label} ({conf*100:.2f}%)")