import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from tensorflow.keras.models import load_model
from src.preprocessing import get_generators
from sklearn.preprocessing import label_binarize
import os

# ---------- CLASS NAMES ----------
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

def evaluate_model():
    # create outputs folder
    os.makedirs("outputs", exist_ok=True)

    # load data
    train_gen, val_gen = get_generators()

    # load model
    model = load_model("models/resnet50_crack_detector.h5")

    # predictions
    preds = model.predict(val_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")

    plt.savefig("outputs/confusion_matrix.png")
    plt.close()

    # ---------- CLASSIFICATION REPORT ----------
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # ---------- ROC CURVE ----------
    y_true_bin = label_binarize(y_true, classes=range(len(CLASS_NAMES)))

    plt.figure(figsize=(10, 7))

    for i in range(len(CLASS_NAMES)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{CLASS_NAMES[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multi-Class)")
    plt.legend()

    plt.savefig("outputs/roc_curve.png")
    plt.close()

    print("\n✅ Saved:")
    print("outputs/confusion_matrix.png")
    print("outputs/roc_curve.png")


if __name__ == "__main__":
    evaluate_model()