"""
author: nn454
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split


tf.config.run_functions_eagerly(True)
print("Eager execution:", tf.executing_eagerly())  # Check if eager execution is enabled




# Define parameters for the model
learning_rate = 0.01
batch_size = 128
n_epochs = 10 # changing from 10 to 20 train longer and 10 for other cases
n_train = 60000
n_test = 10000

# Step 1: Load Fashion MNIST dataset
# Load dataset
(train_images_full, train_labels_full), (test_images, test_labels) = fashion_mnist.load_data()

# Split training data into 80% train, 20% validation # Changing it to 90% train and 10% validation i.e test_size=0.1
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images_full, train_labels_full, test_size=0.2, random_state=42
)


# Normalize pixel values
train_images = train_images.astype('float32') / 255.0
val_images = val_images.astype('float32') / 255.0  # Normalize validation set
test_images = test_images.astype('float32') / 255.0


# Flatten images for logistic regression (28x28 → 784)
train_images = train_images.reshape(-1, 784)
val_images = val_images.reshape(-1, 784)
test_images = test_images.reshape(-1, 784)


# Convert labels to one-hot encoding
train_labels = tf.one_hot(train_labels, depth=10)
val_labels = tf.one_hot(val_labels, depth=10)  # Convert validation labels
test_labels = tf.one_hot(test_labels, depth=10)


# Step 2: Create datasets and iterators
train_data = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).shuffle(len(train_images)).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size)  # New validation dataset
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)


# Step 3: Initialize weights and biases
input_dim = 784  # 28x28 flattened image
output_dim = 10  # 10 classes

w = tf.Variable(tf.random.normal([input_dim, output_dim], stddev=0.01), dtype=tf.float32)
b = tf.Variable(tf.zeros([output_dim]), dtype=tf.float32)

train_acc_history = []
val_acc_history = []


# Step 4: Build model
def logistic_regression(X):
    return tf.matmul(X, w) + b  # Logits before applying softmax

# Step 5: Define loss function (cross-entropy)
def compute_loss(logits, labels):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# Step 6: Define optimizer 
optimizer = tf.optimizers.Adam(learning_rate) # Adam
# optimizer = tf.optimizers.SGD(learning_rate) # SGD
# optimizer = tf.optimizers.RMSprop(learning_rate)  # RMS



# Step 7: Define accuracy calculation
def compute_accuracy(logits, labels):
    preds = tf.argmax(logits, axis=1)
    actuals = tf.argmax(labels, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(preds, actuals), tf.float32))

# Step 8: Train the model
for epoch in range(n_epochs):
    total_loss = 0
    n_batches = 0

    for batch_x, batch_y in train_data:
        with tf.GradientTape() as tape:
            logits = logistic_regression(batch_x)
            loss = compute_loss(logits, batch_y)

        grads = tape.gradient(loss, [w, b])
        optimizer.apply_gradients(zip(grads, [w, b]))

        total_loss += loss.numpy()
        n_batches += 1

    train_accuracy = compute_accuracy(logistic_regression(train_images), train_labels)
    val_accuracy = compute_accuracy(logistic_regression(val_images), val_labels)  # Use new validation set

    train_acc_history.append(train_accuracy.numpy())
    val_acc_history.append(val_accuracy.numpy())



    # print(f"Epoch {epoch + 1}, Loss: {total_loss / n_batches:.4f}, Train Acc: {train_accuracy.numpy():.4f}, Val Acc: {val_accuracy.numpy():.4f}")
    print(f"Epoch {epoch + 1}, Loss: {total_loss / n_batches:.4f}, Train Acc: {train_accuracy.numpy():.4f}, Val Acc: {val_accuracy.numpy():.4f}")


# Step 9: Get final test accuracy
test_logits = logistic_regression(test_images)
test_accuracy = compute_accuracy(test_logits, test_labels)
print(f"Final Test Accuracy: {test_accuracy.numpy():.4f}")

# Step 10: Helper function to plot images
def plot_images(images, y, yhat=None):
    assert len(images) == len(y) == 9

    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape((28, 28)), cmap='binary')

        if yhat is None:
            xlabel = f"True: {y[i]}"
        else:
            xlabel = f"True: {y[i]}, Pred: {yhat[i]}"

        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

# Plot some test images with predictions
images = test_images[:9]
true_labels = tf.argmax(test_labels[:9], axis=1)
predicted_labels = tf.argmax(logistic_regression(images), axis=1)

plot_images(images, true_labels.numpy(), predicted_labels.numpy())

# Step 11: Plot learned weights
def plot_weights():
    w_min, w_max = tf.reduce_min(w).numpy(), tf.reduce_max(w).numpy()

    fig, axes = plt.subplots(3, 4)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        if i < 10:
            image = w[:, i].numpy().reshape((28, 28))
            ax.set_xlabel(f"Weights: {i}")
            ax.imshow(image, vmin=w_min, vmax=w_max, cmap='seismic')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

plot_weights()



plt.figure(figsize=(8, 5))
plt.plot(range(1, n_epochs + 1), train_acc_history, label="Train Accuracy", marker="o")
plt.plot(range(1, n_epochs + 1), val_acc_history, label="Validation Accuracy", linestyle="dashed", marker="s")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train vs Validation Accuracy Over Time")
plt.legend()
plt.grid()
plt.show()

