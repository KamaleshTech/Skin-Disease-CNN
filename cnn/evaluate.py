import os
import glob
import numpy as np
import pandas as pd
import keras

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from preprocessing import preprocess_image
from config import NUM_CLASSES


DATASET_DIR = "dataset"
MODEL_PATH = "cnn_model_trained.keras"
METADATA_FILE = os.path.join(DATASET_DIR, "HAM10000_metadata.csv")

IMAGE_FOLDERS = [
    os.path.join(DATASET_DIR, "HAM10000_images_part_1"),
    os.path.join(DATASET_DIR, "HAM10000_images_part_2")
]

CLASS_NAMES = [
    "akiec",
    "bcc",
    "bkl",
    "df",
    "mel",
    "nv",
    "vasc"
]


# Load metadata
df = pd.read_csv(METADATA_FILE)

# Find images
image_files = []

for folder in IMAGE_FOLDERS:
    image_files.extend(glob.glob(os.path.join(folder, "*.jpg")))

image_map = {
    os.path.splitext(os.path.basename(path))[0]: path
    for path in image_files
}

df["image_path"] = df["image_id"].map(image_map)

# Convert class names to numbers
class_to_index = {
    name: index for index, name in enumerate(CLASS_NAMES)
}

df["label"] = df["dx"].map(class_to_index)


# Create the same 80/10/10 stratified split
test_parts = []

for class_name in CLASS_NAMES:

    class_data = df[df["dx"] == class_name].sample(
        frac=1,
        random_state=42
    )

    total = len(class_data)

    validation_end = int(total * 0.9)

    test_parts.append(
        class_data.iloc[validation_end:]
    )

test_df = pd.concat(test_parts).sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# Load trained model
print("Loading trained model...")

model = keras.models.load_model(MODEL_PATH)


# Predict test images
print("Predicting test images...")

y_true = []
y_pred = []

for _, row in test_df.iterrows():

    image = preprocess_image(row["image_path"])

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image, verbose=0)

    predicted_class = np.argmax(prediction[0])

    y_true.append(row["label"])
    y_pred.append(predicted_class)


# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)

precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0
)

cm = confusion_matrix(
    y_true,
    y_pred
)


# Display results
print()
print("===================================")
print("CNN MODEL EVALUATION")
print("===================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

print()
print("Confusion Matrix:")
print(cm)

print("===================================")