# -*- coding: utf-8 -*-
"""
Created on Sun Jul 31 13:16:38 2022

@author: vikas
"""

input_vector = [1.72, 1.23]
weights_1 = [1.26, 0]
weights_2 = [2.17, 0.32]

# Computing the dot product of input_vector and weights_1
first_indexes_mult = input_vector[0] * weights_1[0]
second_indexes_mult = input_vector[1] * weights_1[1]
dot_product_1 = first_indexes_mult + second_indexes_mult

print(f"The dot product is: {dot_product_1}")

import numpy as np

dot_product_1 = np.dot(input_vector, weights_1)

print(f"The dot product is: {dot_product_1}")

dot_product_2 = np.dot(input_vector, weights_2)

print(f"The dot product is: {dot_product_2}")



#Applying Sigmoid

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def make_prediction(input_vector, weights, bias):
    layer_1 = np.dot(input_vector, weights) + bias
    layer_2 = sigmoid(layer_1)
    return layer_2

prediction = make_prediction(input_vector, weights_1, bias=1)
print(f"The prediction result is: {prediction}")


#Different Input Vector
input_vector = np.array([2, 1.5])
prediction = make_prediction(input_vector, weights_1, bias=1)
print(f"The prediction result is: {prediction}")


target = 1
mse = np.square(prediction - target)
print(f"Prediction: {prediction}; Error: {mse}")



#Role of Gradient Descent
#https://realpython.com/gradient-descent-algorithm-python/

import numpy as np

def gradient_descent(gradient, start, learn_rate, n_iter=50, tolerance=1e-06):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            break
        vector += diff
    return vector


gradient_descent(gradient=lambda v: 2 * v, start=10.0, learn_rate=0.2)

#Impact of Learning Rate
gradient_descent(gradient=lambda v: 2 * v, start=10.0, learn_rate=0.8)
gradient_descent(gradient=lambda v: 2 * v, start=10.0, learn_rate=0.005)




#Impact of Iterations
gradient_descent(gradient=lambda v: 2 * v, start=10.0, learn_rate=0.005,n_iter=100)

gradient_descent(gradient=lambda v: 2 * v, start=10.0, learn_rate=0.005,n_iter=1000)

gradient_descent(gradient=lambda v: 2 * v, start=10.0, learn_rate=0.005,n_iter=2000)




#Gradient Decent in Keras

import tensorflow as tf
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
var = tf.Variable(2.5)

cost = lambda: 2 + var ** 2

# Perform optimization
for _ in range(100):
    sgd.minimize(cost, var_list=[var])

# Extract results
var.numpy()

cost().numpy()






class NeuralNetwork:
    def __init__(self, learning_rate):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
    
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
        
        
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
neural_network.predict(input_vector)


import matplotlib.pyplot as plt

input_vectors = np.array([[3, 1.5],[2, 1],[4, 1.5],[3, 4],[3.5, 0.5],[2, 0.5],[5.5, 1],[1, 1],])

targets = np.array([0, 1, 0, 1, 0, 1, 1, 0])
learning_rate = 0.1
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)
plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")




#OR
input_vectors = np.array([[0, 0],[0, 1],[1, 0],[1, 1],])
targets = np.array([0, 1, 1, 1])
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)
neural_network.predict(input_vectors)



#AND
input_vectors = np.array([[0, 0],[0, 1],[1, 0],[1, 1],])
targets = np.array([0, 0, 0, 1])
neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(input_vectors, targets, 10000)
neural_network.predict(input_vectors)




'''
#pima-indians-diabetes.csv


class NeuralNetwork:
    def __init__(self, learning_rate, shape):
        self.weights = np.array([0 for i  in range(shape)])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2
        return prediction

    def _compute_gradients(self, input_vector, target):
        layer_1 = np.dot(input_vector, self.weights) + self.bias
        layer_2 = self._sigmoid(layer_1)
        prediction = layer_2

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(layer_1)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
    
    def train(self, input_vectors, targets, iterations):
        cumulative_errors = []
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                cumulative_error = 0
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                cumulative_errors.append(cumulative_error)

        return cumulative_errors
        


import pandas as pd

df = pd.read_csv('pima-indians-diabetes.csv')

df.columns

X= df.drop('Result', axis=1).values

Y = df['Result'].values

neural_network = NeuralNetwork(learning_rate)
training_error = neural_network.train(X, Y, 10)

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")


from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X,Y, test_size=0.1)
neural_network = NeuralNetwork(learning_rate, X.shape[1])
training_error = neural_network.train(trainX, trainY, 10000)

pred = neural_network.predict(testX)
pred

plt.plot(training_error)
plt.xlabel("Iterations")
plt.ylabel("Error for all training instances")
plt.savefig("cumulative_error.png")

'''

























