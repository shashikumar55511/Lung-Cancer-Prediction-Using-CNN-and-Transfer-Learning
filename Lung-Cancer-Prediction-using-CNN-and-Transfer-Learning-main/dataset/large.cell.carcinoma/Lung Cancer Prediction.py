# Import necessary libraries
import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing import image

# Main function
def main():
    print("Libraries Imported")

    # Set up paths
    BASE_DIR = "/Users/arunkumaraluru/Downloads/micro/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/dataset"
    train_folder = os.path.join(BASE_DIR, "train")
    test_folder = os.path.join(BASE_DIR, "test")
    validate_folder = os.path.join(BASE_DIR, "valid")

    # Set the image size for resizing
    IMAGE_SIZE = (350, 350)
    batch_size = 8
    OUTPUT_SIZE = 4

    # Initialize image data generators
    train_datagen = ImageDataGenerator(rescale=1.0 / 255, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    # Create the training data generator
    train_generator = train_datagen.flow_from_directory(
        train_folder,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical",
    )

    # Create the validation data generator
    validation_generator = test_datagen.flow_from_directory(
        validate_folder if os.path.exists(validate_folder) else test_folder,
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        color_mode="rgb",
        class_mode="categorical",
    )

    # Calculate steps per epoch and validation steps dynamically
    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = validation_generator.samples // batch_size

    # Set up callbacks
    learning_rate_reduction = ReduceLROnPlateau(
        monitor="val_loss", patience=5, verbose=2, factor=0.5, min_lr=0.000001
    )
    early_stops = EarlyStopping(
        monitor="val_loss", min_delta=0, patience=6, verbose=2, mode="auto"
    )
    checkpointer = ModelCheckpoint(
    filepath="/Users/arunkumaraluru/Downloads/micro/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/models/best_model.weights.h5", 
    verbose=2, 
    save_best_only=True, 
    save_weights_only=True
)

    # Load a pre-trained model and add custom layers
    pretrained_model = Xception(
        weights="imagenet", include_top=False, input_shape=[*IMAGE_SIZE, 3]
    )
    pretrained_model.trainable = False

    model = Sequential([
        pretrained_model,
        GlobalAveragePooling2D(),
        Dense(OUTPUT_SIZE, activation="softmax"),
    ])

    print("Pretrained model summary:")
    pretrained_model.summary()

    print("Final model summary:")
    model.summary()

    # Compile the model
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=50,
        callbacks=[learning_rate_reduction, early_stops, checkpointer],
        validation_data=validation_generator,
        validation_steps=validation_steps,
    )

    print("Final training accuracy =", history.history["accuracy"][-1])
    print("Final validation accuracy =", history.history["val_accuracy"][-1])

    # Save the trained model
    model.save(os.path.join(BASE_DIR, "/Users/arunkumaraluru/Downloads/micro/Lung-Cancer-Prediction-using-CNN-and-Transfer-Learning-main/models/trained_lung_cancer_model.h5"))

    # Display training curves
    display_training_curves(history.history)

    # Test the model on individual images
    test_images = ["sq.png", "ad3.png", "l3.png", "n8.jpg"]
    for img_name in test_images:
        img_path = os.path.join(BASE_DIR, img_name)
        if os.path.exists(img_path):
            predict_image(img_path, model, train_generator.class_indices, IMAGE_SIZE)


def display_training_curves(history):
    """Display loss and accuracy curves."""
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(history["loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(history["accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_image(img_path, model, class_indices, image_size):
    """Predict and visualize the class of an input image."""
    img = load_and_preprocess_image(img_path, image_size)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    class_labels = list(class_indices.keys())
    predicted_label = class_labels[predicted_class]
    confidence = predictions[0][predicted_class] * 100

    print(f"The image {os.path.basename(img_path)} belongs to class: {predicted_label} ({confidence:.2f}%)")
    plt.imshow(image.load_img(img_path, target_size=image_size))
    plt.title(f"Predicted: {predicted_label} ({confidence:.2f}%)")
    plt.axis("off")
    plt.show()


def load_and_preprocess_image(img_path, target_size):
    """Load and preprocess an image for prediction."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array


if __name__ == "__main__":
    main()
