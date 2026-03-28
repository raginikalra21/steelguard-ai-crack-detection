import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing import image
import os
import random

# ---------- CLASS NAMES ----------
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# ---------- LOAD MODEL ----------
model = load_model("models/resnet50_crack_detector.h5")

# ---------- REBUILD FUNCTIONAL MODEL ----------
# Extract layers from your Sequential model
base_model = model.layers[0]   # ResNet50
x = base_model.output

# Recreate the top layers manually (same as training)
for layer in model.layers[1:]:
    x = layer(x)

functional_model = Model(inputs=base_model.input, outputs=x)

# ---------- LAST CONV LAYER ----------
last_conv_layer = base_model.get_layer("conv5_block3_out")

# ---------- GRAD MODEL ----------
grad_model = Model(
    inputs=functional_model.input,
    outputs=[last_conv_layer.output, functional_model.output]
)

# ---------- GRAD-CAM FUNCTION ----------
def get_gradcam(img_array):
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)

    return heatmap, int(pred_index)

# ---------- RUN ----------
def run_gradcam(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    heatmap, pred_idx = get_gradcam(img_array)
    pred_label = CLASS_NAMES[pred_idx]

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(img_path)
    original = cv2.resize(original, (224, 224))

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title(f"Grad-CAM: {pred_label}")
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# ---------- AUTO IMAGE ----------
if __name__ == "__main__":
    folder = "data/raw/dataset/validation/images/scratches"
    img_name = random.choice(os.listdir(folder))
    test_image = os.path.join(folder, img_name)

    print(f"Using image: {test_image}")
    run_gradcam(test_image)