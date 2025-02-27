import time
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    
    #Get indices
    n = train_x.shape[0]
    indices = np.arange(n)
    np.random.shuffle(indices)
    #Send n samples at a time where n = batch_size
    for i in range(0, n, batch_size):
        this_batch = indices[i:i+batch_size]
        yield train_x[this_batch], train_y[this_batch]
        

class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass

class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        
        x = self.forward(x)
        return x * (1 - x)

class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(x)
    
    def derivative(self, x):
        return 1 - np.tanh(x)**2

class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)
    def derivative(self, x):
        return np.where(x > 0, 1, 0)

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x2 = np.exp(x - np.max(x, axis=axis, keepdims=True))
        x3 = x2 / np.sum(x2, axis=axis, keepdims=True)
        return x3
    
    #Did not need to implement for this project because I am
    #using the CrossEntropy loss function for the mnist dataset
    def derivative(self, x: np.ndarray) -> np.ndarray:
        pass
        
class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x
    def derivative(self, x):
        return np.ones_like(x)

class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum((y_true - y_pred) ** 2)
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return y_pred - y_true

class CrossEntropy(LossFunction):
    
    def one_hot_encode(self, y: np.ndarray, num_classes: int) -> np.ndarray:
        one_hot = np.zeros((y.shape[0], num_classes))
        one_hot[np.arange(y.shape[0]), y] = 1
        return one_hot
    
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred_clipped = np.clip(y_pred, 1e-12, 1.0)
        return -np.sum(y_true * np.log(y_pred_clipped), axis=1)
    
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        
        return y_pred - y_true

class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        self.W = np.random.uniform(-np.sqrt(6.0 / (fan_in + fan_out)), np.sqrt(6.0 / (fan_in + fan_out)), (fan_in, fan_out))  # weights
        self.b = np.zeros((1, fan_out))  # biases
    
    
    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """
        send_to_activation = np.dot(h, self.W) + self.b
        self.activations = self.activation_function.forward(send_to_activation)

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """
        
        #If the activation is softmax and the layer is the last layer
        #then you want to use the delta
        if isinstance(self.activation_function, Softmax):
            self.delta = delta.dot(self.W.T)
            dL_dW = h.T.dot(self.delta)
            dL_db = np.sum(self.delta, axis=0, keepdims=True)
        else:
            Z_of_current_layer = np.dot(h, self.W) + self.b
            activation_derivative = self.activation_function.derivative(Z_of_current_layer)
            self.delta = (delta * activation_derivative).dot(self.W.T)
            dL_dW = h.T.dot(delta * activation_derivative)
            dL_db = np.sum(delta * activation_derivative, axis=0, keepdims=True)
            
        
        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        
        #You want to copy x for the first forward pass so that you don't pass the entire list
        #in subsequent forward passes
        layer_output = x
        for layer in self.layers:
            layer_output = layer.forward(layer_output)
        return layer_output

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        
        dl_dw_all = []
        dl_db_all = []
        loss_of_layer_above = loss_grad
        
        for i in reversed(range(len(self.layers))):
            #The First layer does not have a previous layer to get an activation from so you
            #just use the input data
            if i == 0:
                dl_dw, dl_db = self.layers[i].backward(input_data, loss_of_layer_above)
            #Otherwise you want to use the activation of the layer below
            else:
                dl_dw, dl_db = self.layers[i].backward(self.layers[i-1].activations, loss_of_layer_above)
            dl_db_all.insert(0,dl_db)
            dl_dw_all.insert(0,dl_dw)
            loss_of_layer_above = self.layers[i].delta
        return dl_dw_all, dl_db_all

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32) -> Tuple[np.ndarray, np.ndarray]:
        
        """
        Train the multilayer perceptron

        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        
        training_losses = []
        validation_losses = []
        

        for epoch in range(epochs):
            totalLoss = 0.0
            batchCount = 0
            #Go over every batch in training set before testing validation
            for mini_train_x, mini_train_y in batch_generator(train_x, train_y, batch_size):
                
                prediction = self.forward(mini_train_x)
                loss = loss_func.loss(mini_train_y, prediction)
                totalLoss += np.mean(loss)
                gradient = loss_func.derivative(mini_train_y, prediction)
                weights, biases = self.backward(loss_grad=gradient, input_data=mini_train_x)
                batchCount +=1
                for index, layer in enumerate(self.layers):
                    if layer.W.shape == weights[index].shape and layer.b.shape == biases[index].shape:
                        layer.W -= learning_rate * weights[index]
                        layer.b -= learning_rate * biases[index]
                    
            trainingLoss = totalLoss/batchCount
            training_losses.append(trainingLoss)
            predictionV = self.forward(val_x)
            lossV = loss_func.loss(val_y, predictionV)
            validationLoss = np.mean(lossV)
            validation_losses.append(validationLoss)
            
            print("Epoch: {} ---------------------------".format(epoch))
            print("     Training Loss: {}".format(trainingLoss))
            print("     Validation Loss: {}".format(validationLoss))
                
            
        return training_losses, validation_losses
    
