import numpy as np
import keras

from config import INPUT_SHAPE, NUM_CLASSES, DROPOUT_RATE
from preprocessing import preprocess_image


def build_cnn_model():
    model = keras.Sequential([
        keras.layers.Input(shape=INPUT_SHAPE),

        # Data Augmentation
        keras.layers.RandomFlip("horizontal"),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),

        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Conv2D(128, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),

        keras.layers.Flatten(),

        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(DROPOUT_RATE),

        keras.layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    return model


def compile_cnn_model(model):
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )


def save_cnn_model(model, file_path="cnn_model.keras"):
    model.save(file_path)
    print(f"Model saved successfully: {file_path}")


if __name__ == "__main__":
    # Build and compile CNN
    model = build_cnn_model()
    compile_cnn_model(model)

    # Load and preprocess a real HAM10000 image
    image_path = "dataset/HAM10000_images_part_1/ISIC_0024306.jpg"
    image = preprocess_image(image_path)

    # Add batch dimension
    image = np.expand_dims(image, axis=0)

    # Test prediction
    prediction = model.predict(image, verbose=0)

    print("Input shape:", image.shape)
    print("Output shape:", prediction.shape)
    print("Prediction probabilities:", prediction[0])
    print("Predicted class index:", np.argmax(prediction[0]))

    # Save model
    save_cnn_model(model)