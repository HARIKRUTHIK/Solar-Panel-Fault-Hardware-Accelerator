# =============================================================================
# B.Tech Capstone Project: Solar Panel Fault Classification
# School of Electronics Engineering (SENSE), VIT Vellore
# Team: Hari Kruthik SK (Model Dev), Rudresh KP (Optimization), Viswin Kumar RK (Data)
# Backbone: EfficientNetV2-B0 Transfer Learning & Fine-Tuning
# Dataset: 21,660 images across 4 classes (Clean, Dusty, Bird Drop, Physical Damage)
# =============================================================================

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASSES = ['Clean', 'Dusty', 'Bird Drop', 'Physical Damage']

def build_solar_efficientnet_model():
    """Build EfficientNetV2-B0 transfer learning architecture for PV condition diagnosis."""
    base_model = tf.keras.applications.EfficientNetV2B0(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = True

    # Freeze shallow feature extraction layers, fine-tune top stages
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    inputs = keras.Input(shape=(*IMG_SIZE, 3))
    # Augmentation
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)

    # Backbone
    x = base_model(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation='softmax', name="condition_output")(x)

    model = keras.Model(inputs, outputs, name="Solar_Fault_EfficientNetV2B0")
    return model

if __name__ == "__main__":
    model = build_solar_efficientnet_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    model.summary()
    print("[INFO] Solar Fault EfficientNetV2-B0 model constructed successfully.")
