"""
author: nn454
code: lin_reg.py
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



tf.config.run_functions_eagerly(True)

seed_value = sum(ord(c) for c in "Neha")  # Unique seed
tf.random.set_seed(seed_value)
np.random.seed(seed_value)
tf.random.set_seed(seed_value)
np.random.seed(seed_value)

# Generate data: y = 3x + 2 + noise
NUM_EXAMPLES = 10000
X = tf.random.normal([NUM_EXAMPLES]) 
noise = tf.random.normal([NUM_EXAMPLES])  # Add some noise # Gaussian Noise
# Low level noise
# noise = tf.random.normal([NUM_EXAMPLES], stddev=0.1)
# High level noise
# noise = tf.random.normal([NUM_EXAMPLES], stddev=2.0)
# Uniform Noise
# noise = tf.random.uniform([NUM_EXAMPLES], minval=-1.0, maxval=1.0)
# Laplacian Noise
# noise = tf.random.normal([NUM_EXAMPLES], stddev=1.0)
# noise = tf.math.sign(noise) * tf.math.log(1.0 + tf.abs(noise))


y = X * 3 + 2 + noise  # True output

# Initialize trainable variables W and b to 0 
W = tf.Variable(0.5)  # Initializing W to 0
b = tf.Variable(0.5)  # Initializing b to 0
# Small random initial value
# W = tf.Variable(tf.random.normal([]))  
# b = tf.Variable(tf.random.normal([]))

# Training hyperparameters
train_steps = 5000  
initial_learning_rate = 0.01
learning_rate = initial_learning_rate
patience = 100  # Early stopping patience
best_loss = float('inf')
patience_counter = 0
lr_reduction_factor = 0.5 # patience scheduling

# Loss functions
def mse_loss(y, y_predicted):
    return tf.reduce_mean(tf.square(y - y_predicted))

def mae_loss(y, y_predicted):
    return tf.reduce_mean(tf.abs(y - y_predicted))

def huber_loss(y, y_predicted, delta=1.0):
    error = y - y_predicted
    is_small_error = tf.abs(error) <= delta
    squared_loss = 0.5 * tf.square(error)
    linear_loss = delta * (tf.abs(error) - 0.5 * delta)
    return tf.reduce_mean(tf.where(is_small_error, squared_loss, linear_loss))

def hybrid_loss(y, y_predicted, alpha=0.5):
    # alpha controls the trade-off between L1 (MAE) and L2 (MSE) components.
    return alpha * mse_loss(y, y_predicted) + (1 - alpha) * mae_loss(y, y_predicted)

# Select loss function (change this to test different losses)
selected_loss = mse_loss  # Change to mse_loss, mae_loss, huber_loss or hybrid_loss,


# Gradient Descent Training
loss_history = []

for step in range(train_steps):
    # Add noise in data. 
    # Introduce varying noise to data every 500 steps
    # if step % 500 == 0:
    #     noise_stddev = np.random.uniform(0.5, 2.0)  # Varying noise level
    #     noise = tf.random.normal([NUM_EXAMPLES], stddev=noise_stddev)
    #     y = X * 3 + 2 + noise  # Update y with new noise

    with tf.GradientTape() as tape:
        y_pred = W * X + b  # Prediction
        loss_value = selected_loss(y, y_pred)  # Compute loss

    # Compute gradients
    gradients = tape.gradient(loss_value, [W, b])

    # Add noise in your weights. 
    # Add noise to weights every 500 steps
    # if step % 500 == 0:
    #     weight_noise = tf.random.normal(W.shape, stddev=0.05)  # Small noise to weights
    #     bias_noise = tf.random.normal(b.shape, stddev=0.05)  # Small noise to bias
    #     W.assign_add(weight_noise)
    #     b.assign_add(bias_noise)

    # Add noise in your learning rate
    # Add noise to learning rate every 200 steps
    # if step % 200 == 0:
    #     lr_noise = np.random.uniform(0.9, 1.1)  # Small fluctuation (±10%)
    #     learning_rate *= lr_noise  # Adjust learning rate slightly
    #     print(f"Step {step}: Learning rate adjusted with noise to {learning_rate:.6f}")

    
    # Update parameters
    W.assign_sub(learning_rate * gradients[0])
    b.assign_sub(learning_rate * gradients[1])

    # Store loss history
    loss_history.append(loss_value.numpy())

    # Early Stopping
    if loss_value < best_loss:
        best_loss = loss_value
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            learning_rate *= 0.5  # Reduce learning rate by half
            patience_counter = 0  # Reset patience counter
            print(f"Reducing learning rate to {learning_rate:.6f} at step {step}")
            if learning_rate < 1e-6:  # If learning rate is too small, stop training
                print(f"Stopping training as learning rate has reached {learning_rate:.6f}")
                break


    # Print progress every 100 steps
    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss_value.numpy():.4f}, W: {W.numpy():.4f}, b: {b.numpy():.4f}")

# Final Results
print(f"\nFinal Parameters: W = {W.numpy():.4f}, b = {b.numpy():.4f}")

# Plot loss history
plt.figure(figsize=(8, 5))
plt.plot(loss_history, label="Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title(f"Loss Curve - {selected_loss.__name__}")
plt.legend()
plt.show()

# Plot regression line (Corrected with loss function name)
plt.figure(figsize=(8, 5))
plt.scatter(X.numpy(), y.numpy(), color='blue', label="Original Data")  # Original data points
plt.plot(X.numpy(), W.numpy() * X.numpy() + b.numpy(), 'r', 
         label=f"Regression Fit - {selected_loss.__name__}")  # Regression line with loss name
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title(f"Linear Regression Fit ({selected_loss.__name__})")
plt.show()

