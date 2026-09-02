import os
import glob
import numpy as np
import pandas as pd
import keras

from cnn_model import build_cnn_model, compile_cnn_model
from preprocessing import preprocess_image
from config import NUM_CLASSES


# =========================
# Configuration
# =========================

BATCH_SIZE = 16
EPOCHS = 10
RANDOM_STATE = 42

DATASET_DIR = "dataset"
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

CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}


# =========================
# Dataset Generator
# =========================

class HAM10000Sequence(keras.utils.Sequence):

    def __init__(self, dataframe, batch_size=BATCH_SIZE, shuffle=True):
        super().__init__()

        self.dataframe = dataframe.reset_index(drop=True)
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.dataframe))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.dataframe) / self.batch_size))

    def __getitem__(self, index):

        start = index * self.batch_size
        end = min(start + self.batch_size, len(self.dataframe))

        batch = self.dataframe.iloc[start:end]

        images = []
        labels = []

        for _, row in batch.iterrows():

            image = preprocess_image(row["image_path"])

            label = row["label"]

            images.append(image)
            labels.append(label)

        images = np.array(images, dtype=np.float32)

        labels = keras.utils.to_categorical(
            labels,
            num_classes=NUM_CLASSES
        )

        return images, labels

    def on_epoch_end(self):

        if self.shuffle:
            np.random.shuffle(self.indexes)
            self.dataframe = self.dataframe.iloc[
                self.indexes
            ].reset_index(drop=True)


# =========================
# Load Dataset
# =========================

print("Loading HAM10000 metadata...")

df = pd.read_csv(METADATA_FILE)

print("Total records:", len(df))


# =========================
# Find Image Paths
# =========================

print("Finding image files...")

image_files = []

for folder in IMAGE_FOLDERS:
    image_files.extend(
        glob.glob(os.path.join(folder, "*.jpg"))
    )

image_map = {
    os.path.splitext(os.path.basename(path))[0]: path
    for path in image_files
}

df["image_path"] = df["image_id"].map(image_map)

missing_images = df["image_path"].isna().sum()

if missing_images > 0:
    raise ValueError(
        f"{missing_images} images were not found."
    )


# =========================
# Convert Classes to Labels
# =========================

df["label"] = df["dx"].map(CLASS_TO_INDEX)

if df["label"].isna().any():
    raise ValueError("Unknown class found in dataset.")


# =========================
# Train / Validation / Test Split
# =========================

train_parts = []
validation_parts = []
test_parts = []

for class_name in CLASS_NAMES:

    class_data = df[df["dx"] == class_name].sample(
        frac=1,
        random_state=RANDOM_STATE
    )

    total = len(class_data)

    train_end = int(total * 0.8)
    validation_end = int(total * 0.9)

    train_parts.append(
        class_data.iloc[:train_end]
    )

    validation_parts.append(
        class_data.iloc[train_end:validation_end]
    )

    test_parts.append(
        class_data.iloc[validation_end:]
    )


train_df = pd.concat(train_parts).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

validation_df = pd.concat(validation_parts).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)

test_df = pd.concat(test_parts).sample(
    frac=1,
    random_state=RANDOM_STATE
).reset_index(drop=True)


print()
print("Dataset split:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


# =========================
# Create Data Generators
# =========================

train_data = HAM10000Sequence(
    train_df,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_data = HAM10000Sequence(
    validation_df,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_data = HAM10000Sequence(
    test_df,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================
# Build CNN
# =========================

print()
print("Building CNN model...")

model = build_cnn_model()

compile_cnn_model(model)

model.summary()


# =========================
# Training
# =========================

print()
print("Starting CNN training...")

callbacks = [

    keras.callbacks.ModelCheckpoint(
        "cnn_model_trained.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
]


history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=EPOCHS,
    callbacks=callbacks
)


# =========================
# Test Evaluation
# =========================

print()
print("Evaluating model on test data...")

test_loss, test_accuracy = model.evaluate(
    test_data,
    verbose=1
)

print()
print("===================================")
print("CNN TRAINING COMPLETED")
print("===================================")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print("===================================")


# =========================
# Save Final Model
# =========================

model.save("cnn_model_trained.keras")

print()
print("Trained model saved as:")
print("cnn_model_trained.keras")