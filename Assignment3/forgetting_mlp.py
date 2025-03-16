import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import hashlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model



num_tasks_to_run = 10

num_epochs_first_task = 50  # Initial task training
num_epochs_per_task = 20  # Training for each new task

# Learning parameters
learning_rate = 0.001  # Learning rate for the optimizer
batch_size = 10000  # Batch size for training

# Image and model properties
image_sz = 28  # MNIST images are 28x28 pixels
size_input = image_sz * image_sz  # Flattened input size (784 pixels)
size_hidden = 256  # Number of hidden units in each layer
size_output = 10  


def get_seed_from_name(name):
    hash_object = hashlib.sha256(name.encode())  
    seed = int(hash_object.hexdigest(), 16) % (2**32) 
    return seed

random_seed = get_seed_from_name("Neha")
np.random.seed(random_seed)
tf.random.set_seed(random_seed)

print(f"Generated Random Seed from 'Neha': {random_seed}")

# MNIST
def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = x_train.reshape(-1, size_input)  
    x_test = x_test.reshape(-1, size_input)  

    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_mnist()


print(f"MNIST Loaded: Training set {x_train.shape}, Test set {x_test.shape}")


def generate_permuted_tasks(num_tasks=num_tasks_to_run):
    return [np.random.permutation(size_input) for _ in range(num_tasks)]


task_permutations = generate_permuted_tasks(num_tasks_to_run)

print(f"First 10 indices of permutation 1: {task_permutations[0][:10]}")



def apply_permutation(images, permutation):
    return images[:, permutation]


x_train_task1 = apply_permutation(x_train, task_permutations[0])
x_test_task1 = apply_permutation(x_test, task_permutations[0])


print(f"Example of permuted task: Training set {x_train_task1.shape}, Test set {x_test_task1.shape}")

import matplotlib.pyplot as plt


indices = [0, 10, 20, 30]  


permutation = task_permutations[0]

fig, axes = plt.subplots(2, 4, figsize=(10, 5))

for i, idx in enumerate(indices):

    original_image = x_train[idx].reshape(28, 28)
    permuted_image = apply_permutation(x_train, permutation)[idx].reshape(28, 28)

    # Plot original images (first row)
    axes[0, i].imshow(original_image, cmap='gray')
    axes[0, i].set_title(f"Original {idx}")
    axes[0, i].axis("off")

    # Plot permuted images (second row)
    axes[1, i].imshow(permuted_image, cmap='gray')
    axes[1, i].set_title(f"Permuted {idx}")
    axes[1, i].axis("off")


plt.suptitle("Original vs. Permuted MNIST Images", fontsize=14)
plt.tight_layout()
plt.show()


# MLP Model
def create_mlp_model(depth=2, dropout_rate=0.5, optimizer_name="adam", loss_function="nll"):

    model = Sequential()
    
    # Input Layer
    model.add(Dense(size_hidden, activation="relu", input_shape=(size_input,)))
    
    # Hidden Layers
    for _ in range(depth - 1):
        model.add(Dense(size_hidden, activation="relu"))
        model.add(Dropout(dropout_rate))  # Apply dropout

    # Output Layer
    model.add(Dense(size_output, activation="softmax"))

    # Optimizer selection
    optimizers = {"sgd": SGD(learning_rate), "adam": Adam(learning_rate), "rmsprop": RMSprop(learning_rate)}
    optimizer = optimizers.get(optimizer_name.lower(), Adam(learning_rate))  # Default to Adam

    # Loss function selection
    if loss_function == "nll":
        loss = SparseCategoricalCrossentropy()  # Standard NLL loss
    elif loss_function == "l1":
        loss = "mean_absolute_error"  # L1 loss
    elif loss_function == "l2":
        loss = "mean_squared_error"  # L2 loss
    elif loss_function == "l1+l2":
        loss = lambda y_true, y_pred: tf.reduce_mean(tf.abs(y_true - y_pred)) + tf.reduce_mean(tf.square(y_true - y_pred))
    else:
        raise ValueError("Invalid loss function. Choose from: 'nll', 'l1', 'l2', 'l1+l2'.")

    # Compile the model
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])

    return model


R = np.zeros((num_tasks_to_run, num_tasks_to_run))


# Select Task A's permutation
task_A_perm = task_permutations[0]
x_train_A = apply_permutation(x_train, task_A_perm)
x_test_A = apply_permutation(x_test, task_A_perm)

# Create and train the model on Task A
mlp_model = create_mlp_model(depth=2, dropout_rate=0.5, optimizer_name="adam", loss_function="nll")

print("\n Training on Task A (First Permuted MNIST Task)...")
history_A = mlp_model.fit(
    x_train_A, y_train,
    epochs=num_epochs_first_task,
    batch_size=batch_size,
    validation_data=(x_test_A, y_test),
    verbose=1
)

loss_history = []  
loss_history.append(history_A.history['val_loss'])

# Evaluate model performance on Task A
task_A_eval = mlp_model.evaluate(x_test_A, y_test, verbose=1)
task_A_accuracy = task_A_eval[1]  # Extract accuracy score

# Store Task A's accuracy in the results matrix
R[0, 0] = task_A_accuracy  

print(f"\nTask A Final Accuracy: {task_A_accuracy:.4f}")



print("\n Current Results Matrix R (After Task A Training):")
print(R)





print("\n Starting Training on All Tasks...\n")

for task_id in range(1, num_tasks_to_run):  # Start from Task B (task_id=1)
    print(f"\n Training on Task {task_id + 1}...\n")
    
    # Get permuted dataset for the current task
    x_train_task = apply_permutation(x_train, task_permutations[task_id])
    x_test_task = apply_permutation(x_test, task_permutations[task_id])
    
    # Train the model on the current task for 20 epochs
    history = mlp_model.fit(
        x_train_task, y_train,
        epochs=num_epochs_per_task,
        batch_size=batch_size,
        validation_data=(x_test_task, y_test),
        verbose=1
    )
    
    loss_history.append(history.history['val_loss'])

    for test_task in range(task_id + 1):  # Evaluate Task A, B, ..., Current Task
        x_test_eval = apply_permutation(x_test, task_permutations[test_task])
        _, acc = mlp_model.evaluate(x_test_eval, y_test, verbose=0)
        R[task_id, test_task] = acc  # Store accuracy in results matrix

    # Print updated results matrix after training each task
    print(f"\nUpdated Results Matrix R (After Task {task_id + 1} Training):")
    print(R)


# Average Accuracy (ACC)
ACC = np.mean(R[-1])  

# Backward Transfer (BWT) 
BWT = np.mean(R[-1, :-1] - np.diag(R[:-1]))  

# Transfer Backward Weight Transfer (TBWT)
TBWT = np.mean(R[-1, :-1] - R[1:, :-1])  

# Cumulative Backward Weight Transfer (CBWT)
CBWT = np.mean([R[j, i] - R[i, i] for i in range(num_tasks_to_run-1) for j in range(i+1, num_tasks_to_run)])

# Print Metrics
print(f"\nForgetting Metrics:\n ACC: {ACC:.4f}, BWT: {BWT:.4f}, TBWT: {TBWT:.4f}, CBWT: {CBWT:.4f}")



num_tasks = len(loss_history)  

# Plot Validation Loss
plt.figure(figsize=(10, 5))
for task_idx, losses in enumerate(loss_history):
    plt.plot(range(1, len(losses) + 1), losses, label=f"Task {task_idx + 1}")

plt.xlabel("Epochs")
plt.ylabel("Validation Loss")
plt.title("Validation Loss Over Training")
plt.legend()
plt.show()


task_numbers = list(range(1, num_tasks_to_run + 1))
final_accuracies = [R[task_id, task_id] for task_id in range(num_tasks_to_run)]  

# Plot Final Accuracy per Task
plt.figure(figsize=(10, 5))
plt.plot(task_numbers, final_accuracies, marker="o", linestyle="-", label="Final Accuracy")
plt.xlabel("Task Number")
plt.ylabel("Accuracy")
plt.title("Final Accuracy per Task")
plt.legend()
plt.show()


# MLP Model Architecture
plot_model(mlp_model, to_file="mlp_model.png", show_shapes=True, show_layer_names=True)

# Plotting graphs
def plot_grouped_bar_chart(categories, metric_values, title, xlabel):
    metrics = ["ACC", "BWT", "TBWT", "CBWT"]
    bar_width = 0.2
    x = np.arange(len(categories))

    plt.figure(figsize=(10, 6))
    plt.bar(x, metric_values[:, 0], width=bar_width, label="ACC", color="blue")
    plt.bar(x + bar_width, metric_values[:, 1], width=bar_width, label="BWT", color="red")
    plt.bar(x + 2 * bar_width, metric_values[:, 2], width=bar_width, label="TBWT", color="green")
    plt.bar(x + 3 * bar_width, metric_values[:, 3], width=bar_width, label="CBWT", color="purple")

    plt.xlabel(xlabel)
    plt.ylabel("Metric Value")
    plt.title(title)
    plt.xticks(x + 1.5 * bar_width, categories)
    plt.axhline(0, color="black", linewidth=1)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# Data for each setting

# Loss Functions
loss_functions = ["NLL", "L1", "L2", "Hybrid"]
loss_values = np.array([
    [0.7269, -0.2639, 0.1791, -0.1983],
    [0.7096, -0.2846, 0.1657, -0.2106],
    [0.7346, -0.2560, 0.1764, -0.1788],
    [0.7538, -0.2339, 0.1975, -0.1780]
])
plot_grouped_bar_chart(loss_functions, loss_values, "Effect of Loss Functions on Forgetting Metrics", "Loss Functions")

# Dropout Rates
dropout_rates = ["0.0", "0.2", "0.5", "0.6"]
dropout_values = np.array([
    [0.6959, -0.2979, 0.1511, -0.2095],
    [0.6956, -0.2992, 0.1429, -0.1960],
    [0.7538, -0.2339, 0.1975, -0.1780],
    [0.6897, -0.3047, 0.1481, -0.2158]
])
plot_grouped_bar_chart(dropout_rates, dropout_values, "Effect of Dropout on Forgetting Metrics", "Dropout Rate")

# MLP Depth
depths = ["2 Layers", "3 Layers", "4 Layers"]
depth_values = np.array([
    [0.7538, -0.2339, 0.1975, -0.1780],
    [0.7148, -0.2767, 0.1731, -0.2110],
    [0.6732, -0.3233, 0.1414, -0.2379]
])
plot_grouped_bar_chart(depths, depth_values, "Effect of MLP Depth on Forgetting Metrics", "MLP Depth")

# Optimizers
optimizers = ["Adam", "SGD", "RMSprop"]
optimizer_values = np.array([
    [0.7538, -0.2339, 0.1975, -0.1780],
    [0.7144, -0.2775, 0.1738, -0.2138],
    [0.7252, -0.2655, 0.1804, -0.2046]
])
plot_grouped_bar_chart(optimizers, optimizer_values, "Effect of Optimizers on Forgetting Metrics", "Optimizer")

