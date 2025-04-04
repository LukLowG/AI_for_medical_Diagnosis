# Post-training Grad-CAM visualization

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.models import Model


# --- Load Metadata ---
base_dir = "./archive"
data_csv_path = os.path.join(base_dir, "Data_Entry_2017.csv")
subfolders = [f for f in os.listdir(base_dir) if f.startswith("images_")]

metadata = pd.read_csv(data_csv_path)
metadata.columns = metadata.columns.str.strip()

# Rebuild full paths
image_path_map = {}
for sub in subfolders:
    sub_path = os.path.join(base_dir, sub, "images")
    if os.path.exists(sub_path):
        for fname in os.listdir(sub_path):
            image_path_map[fname] = os.path.join(sub_path, fname)

metadata["full_path"] = metadata["Image Index"].map(image_path_map.get)
metadata = metadata[
    metadata["full_path"].notnull() & metadata["full_path"].apply(os.path.exists)
]

# --- Load Model ---
model = load_model("best_model.h5")
print("Model loaded.")


# --- Grad-CAM Function ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# --- Create output directory ---
results_dir = "results/gradcam"
os.makedirs(results_dir, exist_ok=True)

# --- Select image ---
sample_image_path = metadata["full_path"].iloc[0]
sample_image = Image.open(sample_image_path).convert("RGB").resize((224, 224))
sample_array = np.expand_dims(np.array(sample_image) / 255.0, axis=0)

# --- Predict and get top-3 classes ---
predictions = model.predict(sample_array)[0]
num_classes = predictions.shape[0]
class_names = [
    f"Class_{i}" for i in range(num_classes)
]  # fallback if true labels unavailable

pred_df = pd.DataFrame({"Class": class_names, "Probability": predictions})
pred_df = pred_df.sort_values(by="Probability", ascending=False).reset_index(drop=True)

# --- Save raw probabilities to CSV ---
pred_df.to_csv(os.path.join(results_dir, "prediction_probabilities.csv"), index=False)

# --- Generate and save Grad-CAM for top-3 classes ---
for i in range(min(3, len(pred_df))):
    class_name = pred_df.loc[i, "Class"]
    class_index = int(class_name.split("_")[-1])
    prob = pred_df.loc[i, "Probability"]

    heatmap = make_gradcam_heatmap(
        sample_array,
        model,
        last_conv_layer_name="conv5_block16_2_conv",
        pred_index=class_index,
    )

    plt.figure(figsize=(6, 6))
    plt.imshow(sample_image)
    hm = plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.title(f"Grad-CAM for class: {class_name} (p={prob:.2f})")
    plt.axis("off")
    cbar = plt.colorbar(hm, shrink=0.7)
    cbar.set_label("Activation Strength")

    output_path = os.path.join(results_dir, f"gradcam_{class_name}.png")
    plt.savefig(output_path)
    plt.close()

print(f"Top-3 Grad-CAM heatmaps saved to: {results_dir}")
print("Prediction probabilities saved to: prediction_probabilities.csv")
