import tensorflow as tf
from tensorflow.keras.datasets import mnist

(x_train, y_train), _ = mnist.load_data()
x_train = x_train / 255.0

model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                        tf.keras.layers.Dense(10)
                        ])

model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                    metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.save("mnist_model.h5")