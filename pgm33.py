import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Set parameters for visualization
plt.rc('figure', autolayout=True)
plt.rc('image', cmap='magma')

# 1. Define the kernel (Filter)
# Must be reshaped to [filter_height, filter_width, in_channels, out_channels]
kernel = tf.constant([[-1, -1, -1],
                      [-1,  8, -1],
                      [-1, -1, -1]], dtype=tf.float32)
kernel = tf.reshape(kernel, [3, 3, 1, 1])

# 2. Load and Preprocess the image
# Using tf.io.read_file and tf.image.decode_jpeg for raw processing
image_raw = tf.io.read_file('Ganesh.jpg')
image = tf.io.decode_jpeg(image_raw, channels=1)
image = tf.image.resize(image, size=[300, 300])

# Reformat image for convolution: [batch, height, width, channels]
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.expand_dims(image, axis=0)

# 3. Convolution Layer (Filtering)
# tf.nn.conv2d requires float32 and specific 4D shapes
image_filter = tf.nn.conv2d(
    input=image,
    filters=kernel,
    strides=[1, 1, 1, 1], # 2026 standards often use 4-element lists for explicit strides
    padding='SAME',
)

# 4. Activation Layer (Detecting)
# ReLU removes negative values to highlight detected features
image_detect = tf.nn.relu(image_filter)

# 5. Pooling Layer (Condensing)
# tf.nn.pool is a general N-D pooling op; window_shape and strides are spatial (H, W)
image_condense = tf.nn.pool(
    input=image_detect, 
    window_shape=(2, 2),
    pooling_type='MAX',
    strides=(2, 2),
    padding='SAME',
)

# Visualization
plt.figure(figsize=(15, 5))

# Original (Grayscale)
plt.subplot(1, 4, 1)
plt.imshow(tf.squeeze(image), cmap='gray')
plt.title('Original Gray')
plt.axis('off')

# After Convolution
plt.subplot(1, 4, 2)
plt.imshow(tf.squeeze(image_filter))
plt.title('Convolution (Edges)')
plt.axis('off')

# After ReLU
plt.subplot(1, 4, 3)
plt.imshow(tf.squeeze(image_detect))
plt.title('Activation (ReLU)')
plt.axis('off')

# After Max Pooling
plt.subplot(1, 4, 4)
plt.imshow(tf.squeeze(image_condense))
plt.title('Pooling (Condensed)')
plt.axis('off')

plt.show()
