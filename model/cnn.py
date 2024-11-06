import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Download ResNet50 (pre-trained on ImageNet, suitable for image classification)
from tensorflow.keras.applications import ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduces 7x7x2048 output to 2048
x = Dense(512, activation='relu')(x)  # Dense layer for feature processing
x = Dropout(0.5)(x)  # Dropout for regularization to prevent overfitting
x = Dense(1, activation='sigmoid')(x)  # Final layer for binary classification

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Data generators with more augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,         # Added shear augmentation
    zoom_range=0.2,          # Added zoom augmentation
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load the data
train_generator = train_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/val',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    '/home/ubuntu/chest_xray/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_data=val_generator,
    validation_steps=val_generator.samples // val_generator.batch_size,
    epochs=20,  # Increased epochs for better training
    callbacks=callbacks
)

# Save the trained model
model.save('/home/ubuntu/models/pneumonia_model.keras')

# Inference
from tensorflow.keras.models import load_model
import os

# Load the trained model for predictions
model = load_model('/home/ubuntu/models/pneumonia_model.keras')

# Directory containing test images
test_dir = '/content/chest_xray/test'
for root, dirs, files in os.walk(test_dir):
    for file in files:
        if file.endswith(('.jpg', '.jpeg', '.png')):
            test_image_path = os.path.join(root, file)
            img = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.
            img_array = np.expand_dims(img_array, axis=0)

            # Make the prediction
            prediction = model.predict(img_array)
            confidence = float(prediction[0][0])
            result = "Pneumonia" if confidence > 0.5 else "Normal"

            # Print the results
            print(f"Image: {test_image_path}")
            print(f"Prediction: {result} (confidence: {confidence:.2%})")
            print("---")
