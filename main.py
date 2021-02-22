import numpy as np
import pprint
import nnfs
import matplotlib.pyplot as plt
from nnfs.datasets import spiral_data

'''
Pre- Page 42.

inputs = [1.0, 2.0, 3.0, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]

biases = [2, 3, 0.5]

layer_outputs = []


for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0

    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight

    neuron_output += neuron_bias

    layer_outputs.append(neuron_output)


print(layer_outputs)

outputs2 = np.dot(weights, inputs) + biases

print(outputs2)

'''

'''
Page 42-43

inputs = [1.0, 2.0, 3.0, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]

biases = [2, 3, 0.5]

layer_outputs = np.dot(weights, inputs) + biases

print(layer_outputs)

'''

'''
# Page 53

a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a])
b = np.array([b]).T

print(np.dot(a, b))
'''

'''
# Page 56
# Creating a 3 neuron layer just like before, each with 4 inputs, but we have 3 sets of inputs. Notice the number of
# Weights and Biases stay the same, since we only need one weight per input and one bias per neuron.

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
    ]


weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]

biases = [2, 3, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases

pprint.pprint(outputs)
'''

'''
# Chapter 3 Page 61
# Adding another layer!

inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8],
    ]


weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
]

biases = [2, 3, 0.5]


# The following is for the second layer, which receives input from the above layer. Since there you need a weight per
# input, we only have three weights per neuron in this layer since there are three neurons in the previous layer.
weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]
]

biases2 = [-1, 2, -0.5]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

# This is so cool, I feel like a wizard.
'''

'''
# Page 63-65

from nnfs.datasets import spiral_data
nnfs.init()

X, y = spiral_data(samples=100, classes=3)
print(X)
print(y)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='brg')
plt.show()
'''


# Page 66-71
# Dense Layer (Fully Connected)
class Layer_Dense:  # Ahem -- PEP8 not followed with this class name, Harrison.
    # Layer Initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


# Page 96
# Reticulated Linear Activation
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)


# Page 104
# Softmax activation. Use as a classifier and has confidence levels between 0 and 1.
class Activation_Softmax:
    def forward(self, inputs):

        # Get unnormalized probabilities
        # It took me a minute to understand this section here with why we bother to subtract the max of the inputs below
        # Since we will subtract the largest from itself, it will be 0, which in the exponent will = 1. The others will
        # Be a negative number going toward negative infinity, getting closer to 0 in the output of this function
        # This subtraction helps save from overflow states where numbers are too large, and since we normalize these
        # Values with .exp and then dividing them by their group sum, we always end up with probabilities between 0 - 1.

        # More notes here, "axis=1" is used to group the inputs in the horizontal dimension in the matrix. It returns
        # one value per row in the inputs, and with "keepdims" instead of having [1, 2, 3], we get [[1], [2], [3],]
        # We could use "axis=0" if we wanted to get the max from the column.
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities


# The names 'X' and 'y' here are confusing. X is a list that contains both inputs (x and y in this case for a
# scatterplot) and 'y' contains the associated class for the data.
X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)  # 2 input features, 3 neurons

activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)

activation2 = Activation_Softmax()

dense1.forward(X)

activation1.forward(dense1.output)

dense2.forward(activation1.output)

activation2.forward(dense2.output)

print(activation2.output[:5])

