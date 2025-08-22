from __future__ import annotations

import tensorflow as tf


def get_model_keras(num_classes: int = 10, dropout: float = 0.2) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(28, 28, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    x = tf.keras.layers.Conv2D(128, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def build_preprocess(num_classes: int = 10):
    def train_map(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.expand_dims(x, -1)
        y = tf.one_hot(tf.cast(y, tf.int32), num_classes)
        return x, y

    def test_map(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        x = tf.expand_dims(x, -1)
        y = tf.one_hot(tf.cast(y, tf.int32), num_classes)
        return x, y

    return train_map, test_map
