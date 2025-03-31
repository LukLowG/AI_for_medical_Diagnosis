# Chest X-Ray Diagnosis with Deep Learning - Starter Notebook (NIH Chest X-ray14)

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from collections import Counter
from keras.preprocessing.image import ImageDataGenerator

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

# ---------------------------
# Visualize Sample Images
# ---------------------------

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

# ---------------------------
# Visualize Label Distribution
# ---------------------------


all_labels = metadata["Finding Labels"].str.split("|")
label_counter = Counter(label for sublist in all_labels for label in sublist)

plt.figure(figsize=(12, 6))
sns.barplot(x=list(label_counter.values()), y=list(label_counter.keys()))
plt.title("Number of Samples per Finding Label")
plt.xlabel("Number of Images")
plt.ylabel("Condition")
plt.tight_layout()
plt.show()

# ---------------------------
# ImageDataGenerator Setup
# ---------------------------

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
