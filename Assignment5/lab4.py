import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io
from sklearn.model_selection import train_test_split
import os
import cv2


data_path = 'notMNIST_small/'
classes = sorted(os.listdir(data_path))
print('Classes:', classes)

X = []
y = []

label_map = {c: idx for idx, c in enumerate(classes)}

for c in classes:
    class_path = os.path.join(data_path, c)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:  # Check if image loaded correctly
                    img = cv2.resize(img, (28, 28))  # Resize if necessary
                    img = img / 255.0  # Normalize
                    X.append(img)
                    y.append(label_map[c])
            except:
                # If any error (bad file), just skip
                continue

X = np.array(X)
y = np.array(y)

print('Data shape:', X.shape)
print('Labels shape:', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train.astype(np.float32)
X_test = X_test.astype(np.float32)

print('Train:', X_train.shape, y_train.shape)
print('Test:', X_test.shape, y_test.shape)


class BasicGRU_cell(object):
    def __init__(self, input_units, hidden_units, output_units):
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.Wz = tf.Variable(tf.random.normal([self.input_units, self.hidden_units], stddev=0.1))
        self.Uz = tf.Variable(tf.random.normal([self.hidden_units, self.hidden_units], stddev=0.1))
        self.bz = tf.Variable(tf.zeros([self.hidden_units]))

        self.Wr = tf.Variable(tf.random.normal([self.input_units, self.hidden_units], stddev=0.1))
        self.Ur = tf.Variable(tf.random.normal([self.hidden_units, self.hidden_units], stddev=0.1))
        self.br = tf.Variable(tf.zeros([self.hidden_units]))

        self.Wh = tf.Variable(tf.random.normal([self.input_units, self.hidden_units], stddev=0.1))
        self.Uh = tf.Variable(tf.random.normal([self.hidden_units, self.hidden_units], stddev=0.1))
        self.bh = tf.Variable(tf.zeros([self.hidden_units]))

        self.Wo = tf.Variable(tf.random.truncated_normal([self.hidden_units, self.output_units], stddev=0.1))
        self.bo = tf.Variable(tf.random.truncated_normal([self.output_units], stddev=0.1))

    def GRU(self, previous_hidden_state, x):
        z = tf.sigmoid(tf.matmul(x, self.Wz) + tf.matmul(previous_hidden_state, self.Uz) + self.bz)
        r = tf.sigmoid(tf.matmul(x, self.Wr) + tf.matmul(previous_hidden_state, self.Ur) + self.br)
        h_tilde = tf.tanh(tf.matmul(x, self.Wh) + tf.matmul(r * previous_hidden_state, self.Uh) + self.bh)
        h = (1 - z) * previous_hidden_state + z * h_tilde
        return h

    def get_states(self, processed_input, initial_hidden):
        all_hidden_states = tf.scan(self.GRU, processed_input, initializer=initial_hidden)
        return all_hidden_states

    def get_output(self, hidden_state):
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    def get_outputs(self, processed_input, initial_hidden):
        all_hidden_states = self.get_states(processed_input, initial_hidden)
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
        return all_outputs



class BasicMGU_cell(object):
    def __init__(self, input_units, hidden_units, output_units):
        self.input_units = input_units
        self.hidden_units = hidden_units
        self.output_units = output_units

        self.Wf = tf.Variable(tf.random.normal([self.input_units, self.hidden_units], stddev=0.1))
        self.Uf = tf.Variable(tf.random.normal([self.hidden_units, self.hidden_units], stddev=0.1))
        self.bf = tf.Variable(tf.zeros([self.hidden_units]))

        self.Wh = tf.Variable(tf.random.normal([self.input_units, self.hidden_units], stddev=0.1))
        self.Uh = tf.Variable(tf.random.normal([self.hidden_units, self.hidden_units], stddev=0.1))
        self.bh = tf.Variable(tf.zeros([self.hidden_units]))

        self.Wo = tf.Variable(tf.random.truncated_normal([self.hidden_units, self.output_units], stddev=0.1))
        self.bo = tf.Variable(tf.random.truncated_normal([self.output_units], stddev=0.1))

    def MGU(self, previous_hidden_state, x):
        f = tf.sigmoid(tf.matmul(x, self.Wf) + tf.matmul(previous_hidden_state, self.Uf) + self.bf)
        h_tilde = tf.tanh(tf.matmul(x, self.Wh) + tf.matmul(f * previous_hidden_state, self.Uh) + self.bh)
        h = (1 - f) * previous_hidden_state + f * h_tilde
        return h

    def get_states(self, processed_input, initial_hidden):
        all_hidden_states = tf.scan(self.MGU, processed_input, initializer=initial_hidden)
        return all_hidden_states

    def get_output(self, hidden_state):
        output = tf.nn.relu(tf.matmul(hidden_state, self.Wo) + self.bo)
        return output

    def get_outputs(self, processed_input, initial_hidden):
        all_hidden_states = self.get_states(processed_input, initial_hidden)
        all_outputs = tf.map_fn(self.get_output, all_hidden_states)
        return all_outputs



def loss_fn(logits, labels):
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))


def compute_accuracy(logits, labels):
    preds = tf.argmax(logits, axis=1)
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    return tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))


def train_rnn(model_class, model_name, num_trials=3, hidden_units=128, batch_size=128, epochs=10):
    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for trial in range(num_trials):
        print(f'\n=== Trial {trial+1}/{num_trials} for {model_name} ===')
        
        model = model_class(input_units=28, hidden_units=hidden_units, output_units=10)
        optimizer = tf.keras.optimizers.Adam()

        train_loss_curve = []
        test_loss_curve = []

        for epoch in range(epochs):
            idx = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[idx]
            y_train_shuffled = y_train[idx]

            epoch_loss = 0
            epoch_accuracy = 0
            num_batches = int(np.ceil(X_train.shape[0] / batch_size))

            for i in range(num_batches):
                x_batch = X_train_shuffled[i*batch_size:(i+1)*batch_size]
                y_batch = y_train_shuffled[i*batch_size:(i+1)*batch_size]

                x_batch = tf.transpose(x_batch, perm=[1, 0, 2])  # [time_steps, batch_size, input_size]
                initial_hidden = tf.zeros([x_batch.shape[1], hidden_units])

                with tf.GradientTape() as tape:
                    outputs = model.get_outputs(x_batch, initial_hidden)
                    final_output = outputs[-1]
                    loss = loss_fn(final_output, y_batch)

                grads = tape.gradient(loss, [var for var in vars(model).values() if isinstance(var, tf.Variable)])
                optimizer.apply_gradients(zip(grads, [var for var in vars(model).values() if isinstance(var, tf.Variable)]))

                acc = compute_accuracy(final_output, y_batch)
                epoch_loss += loss.numpy()
                epoch_accuracy += acc.numpy()

            epoch_loss /= num_batches
            epoch_accuracy /= num_batches
            train_loss_curve.append(epoch_loss)

            X_test_t = tf.transpose(X_test, perm=[1, 0, 2])
            initial_hidden_test = tf.zeros([X_test_t.shape[1], hidden_units])
            test_outputs = model.get_outputs(X_test_t, initial_hidden_test)
            final_test_output = test_outputs[-1]
            test_loss = loss_fn(final_test_output, y_test)
            test_acc = compute_accuracy(final_test_output, y_test)

            test_loss_curve.append(test_loss.numpy())

            print(f'Epoch {epoch+1}/{epochs} - Train loss: {epoch_loss:.4f} - Train acc: {epoch_accuracy:.4f} - Test acc: {test_acc:.4f}')

        train_losses.append(train_loss_curve)
        test_losses.append(test_loss_curve)
        train_accuracies.append(epoch_accuracy)
        test_accuracies.append(test_acc.numpy())

    avg_train_loss = np.mean(train_losses, axis=0)
    avg_test_loss = np.mean(test_losses, axis=0)

    print(f'\n=== Final {model_name} Results ===')
    print(f'Average Training Accuracy: {np.mean(train_accuracies)*100:.2f}%')
    print(f'Average Test Accuracy: {np.mean(test_accuracies)*100:.2f}%')
    print(f'Average Classification Error: {100 - np.mean(test_accuracies)*100:.2f}%')

    return avg_train_loss, avg_test_loss




gru_train_loss, gru_test_loss = train_rnn(
    BasicGRU_cell,      
    "GRU",              
    num_trials=3,       
    hidden_units=128,   
    epochs=10            
)

mgu_train_loss, mgu_test_loss = train_rnn(
    BasicMGU_cell,       
    "MGU",              
    num_trials=3,       
    hidden_units=128,    
    epochs=10           
)



plt.figure(figsize=(8,5))
plt.plot(gru_train_loss, label='GRU Train Loss')
plt.plot(gru_test_loss, label='GRU Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('GRU Training vs Test Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(mgu_train_loss, label='MGU Train Loss')
plt.plot(mgu_test_loss, label='MGU Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('MGU Training vs Test Loss')
plt.legend()
plt.grid(True)
plt.show()

