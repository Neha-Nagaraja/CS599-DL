# -*- coding: utf-8 -*-
"""
Author:- nn454
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import hashlib
import gc
from tensorflow.keras.datasets import fashion_mnist

def get_seed_from_name(name):
    import hashlib
    hash_object = hashlib.md5(name.encode())
    seed = int(hash_object.hexdigest(), 16) % (2**32)
    return seed

seed = get_seed_from_name("Neha")
tf.random.set_seed(seed)
np.random.seed(seed)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images.astype('float32') / 255.0
test_images  = test_images.astype('float32')  / 255.0

train_images = np.expand_dims(train_images, axis=-1)
test_images  = np.expand_dims(test_images, axis=-1)

train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels  = tf.keras.utils.to_categorical(test_labels, 10)

default_batch_size = 100   
hidden_size        = 100
learning_rate      = 0.01
output_size        = 10
num_epochs         = 4

class CNN:
    def __init__(self, hidden_size, output_size, mode='no_norm', device=None):

        self.mode = mode
        filter_h, filter_w, filter_c, filter_n = 5, 5, 1, 30
        self.v1 = tf.Variable(tf.random.normal([filter_h, filter_w, filter_c, filter_n], stddev=0.1))
        self.g1 = tf.Variable(tf.norm(self.v1))

        self.v2 = tf.Variable(tf.random.normal([14*14*filter_n, hidden_size], stddev=0.1))
        self.g2 = tf.Variable(tf.norm(self.v2))

        self.v3 = tf.Variable(tf.random.normal([hidden_size, output_size], stddev=0.1))
        self.g3 = tf.Variable(tf.norm(self.v3))

        self.b1 = tf.Variable(tf.zeros([filter_n]), dtype=tf.float32)
        self.b2 = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32)
        self.b3 = tf.Variable(tf.zeros([output_size]), dtype=tf.float32)

        self.gamma_bn = tf.Variable(tf.ones([filter_n]), dtype=tf.float32)
        self.beta_bn  = tf.Variable(tf.zeros([filter_n]), dtype=tf.float32)

        self.gamma_ln = tf.Variable(tf.ones([hidden_size]), dtype=tf.float32)
        self.beta_ln  = tf.Variable(tf.zeros([hidden_size]), dtype=tf.float32)

        self.tf_bn1 = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-5)
        self.tf_ln  = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-5)

        self.param_list = [
            self.v1, self.g1, self.b1,
            self.v2, self.g2, self.b2,
            self.v3, self.g3, self.b3,
            self.gamma_bn, self.beta_bn,
            self.gamma_ln, self.beta_ln
        ]

        self.device       = device
        self.size_output  = output_size

    def flatten(self, X, window_h, window_w, window_c, out_h, out_w, stride=1, padding=0):
        X_padded = tf.pad(X, [[0,0],[padding,padding],[padding,padding],[0,0]])
        windows = []
        for y in range(out_h):
            for x in range(out_w):
                window = tf.slice(X_padded, [0, y*stride, x*stride, 0], [-1, window_h, window_w, -1])
                windows.append(window)
        stacked = tf.stack(windows)
        return tf.reshape(stacked, [-1, window_c*window_w*window_h])

    def convolution(self, X, W, b, padding=2, stride=1):
        n, h, w, c = X.shape
        filter_h, filter_w, filter_c, filter_n = W.shape
        out_h = (h + 2*padding - filter_h)//stride + 1
        out_w = (w + 2*padding - filter_w)//stride + 1

        X_flat = self.flatten(X, filter_h, filter_w, filter_c, out_h, out_w, stride, padding)
        W_flat = tf.reshape(W, [filter_h*filter_w*filter_c, filter_n])
        z = tf.matmul(X_flat, W_flat) + b
        return tf.transpose(tf.reshape(z, [out_h, out_w, n, filter_n]), [2,0,1,3])

    def relu(self, X):
        return tf.maximum(X, 0)

    def max_pool(self, X, pool_h, pool_w, padding=0, stride=2):
        n,h,w,c = X.shape
        out_h = (h+2*padding - pool_h)//stride + 1
        out_w = (w+2*padding - pool_w)//stride + 1
        X_flat = self.flatten(X, pool_h, pool_w, c, out_h, out_w, stride, padding)
        pool   = tf.reduce_max(tf.reshape(X_flat, [out_h, out_w, n, pool_h*pool_w, c]), axis=3)
        return tf.transpose(pool, [2,0,1,3])

    def affine(self, X, W, b):
        n = X.shape[0]
        X_flat = tf.reshape(X, [n, -1])
        return tf.matmul(X_flat, W) + b

    def weight_norm_custom(self, v, g):
        v_norm = tf.norm(v)
        return (g / v_norm)*v

    def batch_norm_custom(self, x, gamma, beta, eps=1e-5):
        mean = tf.reduce_mean(x, axis=0)
        var  = tf.reduce_mean(tf.square(x - mean), axis=0)
        x_hat= (x - mean)/tf.sqrt(var + eps)
        return gamma*x_hat + beta

    def layer_norm_custom(self, x, gamma, beta, eps=1e-5):
        # x shape => [batch_size, features]
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        var  = tf.reduce_mean(tf.square(x - mean), axis=1, keepdims=True)
        x_hat= (x - mean)/tf.sqrt(var + eps)
        return gamma*x_hat + beta

    def compute_output(self, X):
        if self.mode == "wn_custom":
            W1 = self.weight_norm_custom(self.v1, self.g1)
        else:
            W1 = self.v1
        conv_layer = self.convolution(X, W1, self.b1, padding=2, stride=1)

        if self.mode == "bn_custom":
            conv_layer = self.batch_norm_custom(conv_layer, self.gamma_bn, self.beta_bn)
        elif self.mode == "bn_tf":
            conv_layer = self.tf_bn1(conv_layer, training=True)

        conv_act = self.relu(conv_layer)
        conv_pool= self.max_pool(conv_act, 2, 2, padding=0, stride=2)

        if self.mode == "wn_custom":
            W2 = self.weight_norm_custom(self.v2, self.g2)
        else:
            W2 = self.v2
        affine_out = self.affine(conv_pool, W2, self.b2)

        if self.mode == "ln_custom":
            affine_out = self.layer_norm_custom(affine_out, self.gamma_ln, self.beta_ln)
        elif self.mode == "ln_tf":
            affine_out = self.tf_ln(affine_out, training=True)

        affine_act = self.relu(affine_out)

        if self.mode == "wn_custom":
            W3 = self.weight_norm_custom(self.v3, self.g3)
        else:
            W3 = self.v3
        final_out = self.affine(affine_act, W3, self.b3)

        return final_out

    def forward(self, X):
        return self.compute_output(X)

    def loss(self, y_pred, y_true):
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_tf,
                                                                      labels=y_true_tf))

    def backward(self, X, y, optimizer):
        with tf.GradientTape() as tape:
            preds = self.forward(X)
            current_loss = self.loss(preds, y)
        # Add TF BN or LN variables if relevant
        train_vars = self.param_list[:]
        if self.mode == "bn_tf":
            train_vars += self.tf_bn1.trainable_variables
        elif self.mode == "ln_tf":
            train_vars += self.tf_ln.trainable_variables

        grads = tape.gradient(current_loss, train_vars)
        optimizer.apply_gradients(zip(grads, train_vars))


def train_model(mode="no_norm", epochs=4, batch_size=100):

    tf.keras.backend.clear_session()
    gc.collect()

    model = CNN(hidden_size, output_size, mode=mode)

    # Momentum-based SGD
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)

    train_loss_list = []
    train_acc_list  = []
    test_acc_list   = []

    for epoch in range(epochs):
        train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))\
            .shuffle(1000).batch(batch_size)

        epoch_loss, epoch_acc = 0.0, 0.0
        nb = 0

        for Xb, yb in train_ds:
            preds = model.forward(Xb)
            loss_val = model.loss(preds, yb).numpy()
            acc_val  = accuracy_function(preds, yb).numpy()

            model.backward(Xb, yb, optimizer)

            epoch_loss += loss_val
            epoch_acc  += acc_val
            nb += 1

        avg_loss = epoch_loss / nb
        avg_acc  = (epoch_acc / nb)*100
        train_loss_list.append(avg_loss)
        train_acc_list.append(avg_acc)

        test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size)
        test_acc_total, test_nb = 0.0, 0
        for Xt, yt in test_ds:
            p = model.forward(Xt)
            test_acc_total += accuracy_function(p, yt).numpy()
            test_nb += 1
        epoch_test_acc = (test_acc_total / test_nb)*100
        test_acc_list.append(epoch_test_acc)

        print(f"[{mode}] (bs={batch_size}) Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, "
              f"TrainAcc: {avg_acc:.2f}%, TestAcc: {epoch_test_acc:.2f}%")

    return train_loss_list, train_acc_list, test_acc_list

def accuracy_function(yhat, ytrue):
    return tf.reduce_mean(tf.cast(tf.equal(tf.argmax(yhat,1), tf.argmax(ytrue,1)), tf.float32))

modes_all  = ["no_norm", "bn_custom", "bn_tf", "wn_custom", "ln_custom", "ln_tf"]
labels_all = ["NoNorm", "BN_Cust", "BN_TF", "WN_Cust", "LN_Cust", "LN_TF"]
colors_all = ["red","blue","green","orange","purple","gray"]

results_train_loss = {}
results_train_acc  = {}
results_test_acc   = {}

print("====== Running All 6 Modes with batch_size=100 and Momentum SGD ======\n")
for i,mode in enumerate(modes_all):
    print(f"==== Mode: {mode} ====")
    tloss, tacc, ttest = train_model(mode, epochs=num_epochs, batch_size=default_batch_size)
    results_train_loss[mode] = tloss
    results_train_acc[mode]  = tacc
    results_test_acc[mode]   = ttest

def plot_line_graph(title, x_label, y_label, data_dict, modes_list, labels_list, colors_list):
    plt.figure(figsize=(8,6))
    for i, m in enumerate(modes_list):
        plt.plot(range(1, len(data_dict[m])+1), data_dict[m], label=labels_list[i], color=colors_list[i])
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid()
    plt.show()

# A) BN group => no_norm, bn_custom, bn_tf
bn_modes   = ["no_norm", "bn_custom", "bn_tf"]
bn_labels  = ["NoNorm", "BN_Cust", "BN_TF"]
bn_colors  = ["red","blue","green"]

plot_line_graph("BN Group - Training Accuracy vs Epoch",
                "Epoch", "Train Accuracy (%)",
                results_train_acc, bn_modes, bn_labels, bn_colors)

plot_line_graph("BN Group - Test Accuracy vs Epoch",
                "Epoch", "Test Accuracy (%)",
                results_test_acc, bn_modes, bn_labels, bn_colors)

# B) LN group => no_norm, ln_custom, ln_tf
ln_modes   = ["no_norm", "ln_custom", "ln_tf"]
ln_labels  = ["NoNorm","LN_Cust","LN_TF"]
ln_colors  = ["red","purple","gray"]

plot_line_graph("LN Group - Training Accuracy vs Epoch",
                "Epoch", "Train Accuracy (%)",
                results_train_acc, ln_modes, ln_labels, ln_colors)

plot_line_graph("LN Group - Test Accuracy vs Epoch",
                "Epoch", "Test Accuracy (%)",
                results_test_acc, ln_modes, ln_labels, ln_colors)

# C) WN group => no_norm, wn_custom
wn_modes   = ["no_norm","wn_custom"]
wn_labels  = ["NoNorm","WN_Cust"]
wn_colors  = ["red","orange"]

plot_line_graph("WN Group - Training Accuracy vs Epoch",
                "Epoch", "Train Accuracy (%)",
                results_train_acc, wn_modes, wn_labels, wn_colors)

plot_line_graph("WN Group - Test Accuracy vs Epoch",
                "Epoch", "Test Accuracy (%)",
                results_test_acc, wn_modes, wn_labels, wn_colors)


plot_line_graph("Test Accuracy vs Epoch (All 6 Modes)",
                "Epoch", "Test Accuracy (%)",
                results_test_acc, modes_all, labels_all, colors_all)

final_accs = []
for m in modes_all:
    final_accs.append( results_test_acc[m][-1] )  # last epoch's test acc

plt.figure(figsize=(8,6))
plt.bar(labels_all, final_accs, color=colors_all)
plt.title("Final Test Accuracy (All 6 Modes)")
plt.ylabel("Test Accuracy (%)")
plt.grid(axis='y')
plt.show()

for i,m in enumerate(modes_all):
    print(f"{labels_all[i]} => Final Test Acc: {final_accs[i]:.2f}%")
