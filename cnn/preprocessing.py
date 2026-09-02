import cv2


IMAGE_SIZE = (224, 224)


def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Unable to read image: {image_path}")

    # Resize image
    image = cv2.resize(image, IMAGE_SIZE)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize pixel values from 0-255 to 0-1
    image = image.astype("float32") / 255.0

    return image