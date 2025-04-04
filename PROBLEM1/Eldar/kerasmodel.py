import tensorflow as tf
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras import layers
from keras import ops
from keras import optimizers  # Add this import for optimizer
import pandas as pd


def create_product_recognition_model(num_classes=26, img_height=224, img_width=224):
    """
    Create a product recognition model using MobileNetV3Small
    
    Args:
        num_classes: Number of product categories to classify
        img_height: Input image height
        img_width: Input image width
    """
    # Create the base model with the exact parameters
    base_model = keras.applications.MobileNetV3Small(
        input_shape=(img_height, img_width, 3),
        alpha=1.0,                  # Standard width multiplier
        minimalistic=False,         # Use the full version for better accuracy
        include_top=False,          # Remove the top classification layer
        weights="imagenet",         # Use pre-trained weights
        input_tensor=None,          # Use default input
        classes=1000,               # Not used since include_top is False
        pooling="avg",              # Global average pooling
        dropout_rate=0.2,           # Keep default dropout rate
        classifier_activation="softmax",  # Not used since include_top is False
        include_preprocessing=True,  # Include preprocessing layer
        name="MobileNetV3Small"     # Default name
    )
    
    # Freeze early layers for transfer learning
    for layer in base_model.layers[:-15]:
        layer.trainable = False
    
    # Build the complete model with proper preprocessing/augmentation
    model = keras.Sequential([
        # Data Augmentation Layers
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(factor=0.1, fill_mode="reflect"),
        layers.RandomGrayscale(factor=0.2),
        layers.RandomBrightness(factor=0.2, value_range=(0, 255)),
        layers.RandomSharpness(factor=0.2, value_range=(0, 255)),
        layers.RandomHue(factor=0.1, value_range=(0, 255)),
        layers.RandomContrast(factor=0.2),
        
        # Optional additional augmentations:
        # layers.RandAugment(value_range=(0, 255), num_ops=2, factor=0.3),
        # layers.MixUp(alpha=0.2),
        # layers.AutoContrast(value_range=(0, 255)),
        
        # Base model
        base_model,
        
        # Classification head
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),  # Additional dropout for regularization
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compile with appropriate loss and metrics
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
