from src.preprocessing import get_generators
from src.model import build_model
from src.utils import get_class_weights
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def train_model():
    # Load data
    train_gen, val_gen = get_generators()

    # Compute class weights for imbalance
    class_weights = get_class_weights(train_gen)

    # Build model
    model = build_model()

    # Callbacks
    early_stop = EarlyStopping(
        monitor="val_accuracy",
        patience=5,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        "models/best_resnet50_crack_detector.h5",
        monitor="val_accuracy",
        save_best_only=True
    )

    # Train
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        class_weight=class_weights,
        callbacks=[early_stop, checkpoint]
    )

    # Save final model
    model.save("models/resnet50_crack_detector.h5")

    return history, model


if __name__ == "__main__":
    history, model = train_model()