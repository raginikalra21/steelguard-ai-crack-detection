import numpy as np
import tensorflow as tf
import cv2

def get_gradcam(model, img_array):
    # inner CNN backbone
    base_model = model.get_layer("resnet50")

    # create feature extractor up to last conv layer
    conv_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=base_model.get_layer("conv5_block3_out").output
    )

    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs = conv_model(img_tensor)
        tape.watch(conv_outputs)

        # forward pass through classifier head manually
        x = model.layers[1](conv_outputs)   # GlobalAveragePooling2D
        x = model.layers[2](x)              # Dense
        x = model.layers[3](x)              # Dropout
        predictions = model.layers[4](x)    # Final Dense

        loss = predictions[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy()


def overlay_heatmap(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)

    return result