"""Chest X-Ray Diagnosis with Deep Learning - Starter Notebook (NIH Chest X-ray14)"""

import os
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image


from keras.preprocessing.image import ImageDataGenerator
from keras.applications import DenseNet121
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import tensorflow as tf

# Paths
base_dir = "./archive"
subfolders = [f for f in os.listdir(base_dir) if f.startswith("images_")]
data_csv_path = os.path.join(base_dir, "Data_Entry_2017.csv")

# Load metadata and clean column names
metadata = pd.read_csv(data_csv_path)
metadata.columns = metadata.columns.str.strip()
print("Total entries:", len(metadata))

# Map filenames to their full paths across subfolders
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
print("Valid image paths:", len(metadata))


# Visualize Sample Images


if len(metadata) >= 3:
    sample_df = metadata.sample(3, random_state=42).reset_index(drop=True)
    plt.figure(figsize=(15, 5))
    for i, row in sample_df.iterrows():
        img = Image.open(row["full_path"])
        plt.subplot(1, 3, i + 1)
        plt.imshow(img.convert("L"), cmap="gray")
        plt.title(row["Finding Labels"])
        plt.axis("off")
    plt.suptitle("Sample Chest X-Ray Images with Labels")
    plt.tight_layout()
    plt.show()
else:
    print("Not enough valid images to display samples.")


# Visualize Label Distribution


all_labels = metadata["Finding Labels"].str.split("|")
label_counter = Counter(label for sublist in all_labels for label in sublist)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(label_counter.values()), y=list(label_counter.keys()))
plt.title("Number of Samples per Finding Label")
plt.xlabel("Number of Images")
plt.ylabel("Condition")
plt.tight_layout()
plt.show()

# ImageDataGenerator Setup


metadata["Finding Labels"] = metadata["Finding Labels"].apply(lambda x: x.split("|")[0])

datagen = ImageDataGenerator(rescale=1.0 / 255, validation_split=0.2)

train_gen = datagen.flow_from_dataframe(
    dataframe=metadata,
    x_col="full_path",
    y_col="Finding Labels",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="training",
)

val_gen = datagen.flow_from_dataframe(
    dataframe=metadata,
    x_col="full_path",
    y_col="Finding Labels",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    subset="validation",
)


# Model Definition - DenseNet121


num_classes = len(train_gen.class_indices)

base_model = DenseNet121(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
predictions = Dense(num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Model Training


checkpoint = ModelCheckpoint(
    "best_model.h5", save_best_only=True, monitor="val_loss", mode="min"
)

history = model.fit(
    train_gen, validation_data=val_gen, epochs=5, callbacks=[checkpoint]
)

print("Training complete. Best model saved as 'best_model.h5'")

# Accuracy / Loss Plots

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Acc")
plt.plot(history.history["val_accuracy"], label="Val Acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# Grad-Cam Visualization


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


# Pick a sample image
sample_image_path = metadata["full_path"].iloc[0]
sample_image = Image.open(sample_image_path).convert("RGB").resize((224, 224))
sample_array = np.expand_dims(np.array(sample_image) / 255.0, axis=0)

heatmap = make_gradcam_heatmap(
    sample_array, model, last_conv_layer_name="conv5_block16_2_conv"
)

# Plot heatmap over image
plt.figure(figsize=(6, 6))
plt.imshow(sample_image)
plt.imshow(heatmap, cmap="jet", alpha=0.4)
plt.title("Grad-CAM Overlay")
plt.axis("off")
plt.show()
