import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST
(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Digit you want to extract
desired_digit = 7

# Find all indexes of that digit
indices = np.where(y_test == desired_digit)[0]

# Print first 5 indexes found
print(f"Found {len(indices)} samples for digit {desired_digit}")
print("Sample indices:", indices[:5])

# Pick the first one (you can pick any)
index = indices[0]
digit_image = x_test[index]

# Show the image so we verify it's an actual 8
plt.imshow(digit_image, cmap='gray')
plt.title(f"Digit {desired_digit} at index {index}")
plt.axis('off')
plt.savefig("input_digit_preview.png")
plt.show()

# Normalize and save
digit_image = digit_image / 255.0
np.save('digit_input.npy', digit_image)

print(f"Saved digit {desired_digit} (from index {index}) to digit_input.npy")