import numpy as np
import matplotlib.pyplot as plt
import sys
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear(x):
    return x


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


# 实现了具有两个隐层的神经网络，用于预测回归问题
class NeuralNetwork(object):
    def __init__(self, input_nodes, hidden_1_nodes, hidden_2_nodes, output_nodes, learning_rate):
        # 设定每层的节点个数
        self.input_nodes = input_nodes
        self.hidden_1_nodes = hidden_1_nodes
        self.hidden_2_nodes = hidden_2_nodes
        self.output_nodes = output_nodes

        # 权重初始化
        # 使用零均值单位方差的正态分布值
        # 初始值位于0附近的好处是此时Sigmoid函数的梯度将不会是0，不会遭遇梯度死亡的问题。
        self.weights_i_h1 = np.random.normal(0.0, self.input_nodes ** -0.5, size=(self.input_nodes, self.hidden_1_nodes))
        self.weights_h1_h2 = np.random.normal(0.0, self.hidden_1_nodes ** -0.5, size=(self.hidden_1_nodes, self.hidden_2_nodes))
        self.weights_h2_o = np.random.normal(0.0, self.hidden_2_nodes ** -0.5, size=(self.hidden_2_nodes, self.output_nodes))

        self.alpha = learning_rate

        # Activation Function
        self.activation_h1 = sigmoid
        self.activation_h2 = sigmoid
        self.activation_o = linear

        # Derivative of Activation Function
        self.activation_prime_h1 = lambda x: x * (1 - x)
        self.activation_prime_h2 = lambda x: x * (1 - x)
        self.activation_prime_o = lambda x: 1

    def train(self, features, targets):

        n_records = features.shape[0]
        delta_weights_i_h1 = np.zeros(self.weights_i_h1.shape)
        delta_weights_h1_h2 = np.zeros(self.weights_h1_h2.shape)
        delta_weights_h2_o = np.zeros(self.weights_h2_o.shape)
        for X, y in zip(features, targets):
            # Forward Feed
            # Input → Hidden 1
            h1_in = np.dot(X, self.weights_i_h1)
            h1_out = self.activation_h1(h1_in)

            # Hidden 1 → Hidden 2
            h2_in = np.dot(h1_out, self.weights_h1_h2)
            h2_out = self.activation_h2(h2_in)

            # Hidden 2 → Output
            o_in = np.dot(h2_out, self.weights_h2_o)
            o_out = self.activation_o(o_in)

            # Back Propagation
            o_error_term = (y - o_out) * self.activation_prime_o(o_out)
            h2_error_term = np.dot(self.weights_h2_o, o_error_term) * self.activation_prime_h2(h2_out)
            h1_error_term = np.dot(self.weights_h1_h2, h2_error_term) * self.activation_prime_h1(h1_out)

            delta_weights_h2_o += self.alpha * o_error_term * h2_out[:, None]
            delta_weights_h1_h2 += self.alpha * h2_error_term * h1_out[:, None]
            delta_weights_i_h1 += self.alpha * h1_error_term * X[:, None]

        # Update the weights
        self.weights_h2_o += delta_weights_h2_o / n_records
        self.weights_h1_h2 += delta_weights_h1_h2 / n_records
        self.weights_i_h1 += delta_weights_i_h1 / n_records

    def run(self, features):

        h1_in = np.dot(features, self.weights_i_h1)
        h1_out = self.activation_h1(h1_in)

        # Hidden 1 → Hidden 2
        h2_in = np.dot(h1_out, self.weights_h1_h2)
        h2_out = self.activation_h2(h2_in)

        # Hidden 2 → Output
        o_in = np.dot(h2_out, self.weights_h2_o)
        o_out = self.activation_o(o_in)

        return o_out


# 数据预处理
rides = pd.read_csv('Bike-Sharing-Dataset/hour.csv')
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)
fields_to_drop = ['instant', 'dteday', 'season', 'weathersit',
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']

# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std

test_data = data[-21*24:]
# Now remove the test data from the data set
data = data[:-21*24]
# Separate the data into features and targets
target_fields = ['cnt', 'casual', 'registered']
features, targets = data.drop(target_fields, axis=1), data[target_fields]
test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

train_features, train_targets = features[:-60*24], targets[:-60*24]
val_features, val_targets = features[-60*24:], targets[-60*24:]


# Training Parameters
iterations = 1000
learning_rate = 0.25
n_hidden_1_nodes = 5
n_hidden_2_nodes = 5
n_output_nodes = 1
n_input_nodes = train_features.shape[1]
batch_size = 256

# Create the Neural Net
network = NeuralNetwork(n_input_nodes, n_hidden_1_nodes, n_hidden_2_nodes, n_output_nodes, learning_rate)

losses = {'train': [], 'validation': []}
for ii in range(iterations):
    # Go through a random batch of 128 records from the training data set
    # Mini-Batch 方法
    batch = np.random.choice(train_features.index, size=batch_size)
    X, y = train_features.iloc[batch].values, train_targets.iloc[batch]['cnt']

    network.train(X, y)

    # Printing out the training progress
    train_loss = MSE(network.run(train_features).T, train_targets['cnt'].values)
    val_loss = MSE(network.run(val_features).T, val_targets['cnt'].values)
    sys.stdout.write("\rProgress: {:2.1f}".format(100 * ii / float(iterations))
                     + "% ... Training loss: " + str(train_loss)[:5]
                     + " ... Validation loss: " + str(val_loss)[:5])
    sys.stdout.flush()

    losses['train'].append(train_loss)
    losses['validation'].append(val_loss)


plt.figure(figsize=(15, 5))
plt.plot(losses['train'], label='Training loss')
plt.plot(losses['validation'], label='Validation loss')
plt.legend()
plt.grid(True)
plt.ylim([0, 0.8])
plt.title("Training / Validation Loss")
plt.show()
