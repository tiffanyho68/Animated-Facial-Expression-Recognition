import cv2
import numpy as np
import tensorflow as tf
from keras import layers, models
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

# Preprocess image to ensure correct format
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is not None:
        image = cv2.resize(image, (48, 48))
        image = image / 255.0
    else:
        print(f"Error loading image at path: {image_path}")
        return None

    return image

# Predict emotion from image
def predict_emotion(model, image_path, emotion_classes):
    preprocessed_image = preprocess_image(image_path)

    if preprocessed_image is not None:
        input_data = np.expand_dims(np.expand_dims(preprocessed_image, axis=0), axis=-1)
        prediction = model.predict(input_data)
        emotion_label = np.argmax(prediction)
        predicted_emotion = emotion_classes[emotion_label]
    else:
        predicted_emotion = "Error loading image"

    return predicted_emotion

# Create convolutional neural network (CNN) layers
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001), input_shape=(48, 48, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
num_classes = 3  # number of emotion classes
model.add(layers.Dense(num_classes, activation='softmax'))

data_dir = Path("/Users/tiffanyho/Desktop/CS Projects/CS 4391 Project/training images")
# Get a list of emotion classes (directories)
emotion_classes = [emotion for emotion in os.listdir(data_dir) if (data_dir / emotion).is_dir()]
emotion_classes.sort()  # Ensure consistent order

# Create lists to store image paths and corresponding labels
image_paths = []
labels = []

# Loop through each emotion class
for i, emotion in enumerate(emotion_classes):
    class_dir = data_dir / emotion
    class_images = [f for f in class_dir.glob("*.jpeg") if f.is_file()]

    # Assign labels based on emotion class index
    labels.extend([i] * len(class_images))

    # Extend image_paths with the corresponding image paths and convert Path objects to strings
    image_paths.extend([str(image) for image in class_images])

# Print unique values in labels to investigate unexpected values
unique_labels = set(labels)

# Split the data into training and validation sets with a proper test_size
X_train_paths, X_val_paths, y_train, y_val = train_test_split(
    image_paths, labels, test_size=0.7, random_state=42, stratify=labels
)

# Load and preprocess images with correct input shape
X_train = np.array([preprocess_image(path) for path in X_train_paths])
X_val = np.array([preprocess_image(path) for path in X_val_paths])

# Reshape input data to include batch size and single channel
X_train = X_train.reshape(-1, 48, 48, 1)
X_val = X_val.reshape(-1, 48, 48, 1)

# Convert labels to one-hot encoded format
y_train = tf.keras.utils.to_categorical(y_train, num_classes=num_classes)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=num_classes)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=58, batch_size=32, validation_data=(X_val, y_val))

# Create an ImageDataGenerator for data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Fit the ImageDataGenerator to the training data
datagen.fit(X_train)

# Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Train the model with data augmentation and early stopping
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=20, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Predict emotions on test images
test_image_paths = [
    "/Users/tiffanyho/Desktop/CS Projects/CS 4391 Project/test images/angry eric.jpeg",
    "/Users/tiffanyho/Desktop/CS Projects/CS 4391 Project/test images/angry tina.jpeg",
    "/Users/tiffanyho/Desktop/CS Projects/CS 4391 Project/test images/happy kyle.jpeg",
    "/Users/tiffanyho/Desktop/CS Projects/CS 4391 Project/test images/happy linda.jpeg",
    "/Users/tiffanyho/Desktop/CS Projects/CS 4391 Project/test images/sad bob.jpeg",
    "/Users/tiffanyho/Desktop/CS Projects/CS 4391 Project/test images/sad stan.jpeg",
]

for test_path in test_image_paths:
    predicted_emotion = predict_emotion(model, test_path, emotion_classes)
    print(f"Predicted emotion for {test_path}: {predicted_emotion}")