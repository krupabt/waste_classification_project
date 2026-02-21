# Waste Classification Project

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Dataset paths
train_dir = "dataset/train"
validation_dir = "dataset/validation"
test_dir = "dataset/test"

# Image preprocessing
train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=2,
    class_mode='categorical'
)

# Load validation data
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(224, 224),
    batch_size=2,
    class_mode='categorical'
)

# Load test data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=2,
    class_mode='categorical',
    shuffle=False
)

# Load VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))

# Build model
model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=5
)

# Plot accuracy
plt.plot(history.history['accuracy'])
plt.title("Model Accuracy")
plt.show()

