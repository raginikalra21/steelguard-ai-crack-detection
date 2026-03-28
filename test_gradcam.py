import numpy as np
import cv2
from tensorflow.keras.models import load_model
from src.explainability import get_gradcam, overlay_heatmap

model = load_model("models/resnet50_crack_detector.h5")

img_path = "data/processed/validation/crack/scratches_241.jpg"

img = cv2.imread(img_path)
img = cv2.resize(img, (224, 224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

heatmap = get_gradcam(model, img)

result = overlay_heatmap(img_path, heatmap)

cv2.imwrite("outputs/gradcam_result.jpg", result)

print("✅ Grad-CAM saved at outputs/gradcam_result.jpg")