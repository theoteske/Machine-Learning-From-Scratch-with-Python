### Layers: this file conains the different kind of layers used to build neural networks

import numpy as np
import activations

class InputLayer:
    """Input Layer: in our implementation, the first layer will always be the input layer. So the input is seen as a layer"""
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def forward(self, inputs):
        self.output = inputs
        
    def backward(self, d_output, learning_rate=0.1):
        return d_output

class Dense:
    """Dense, fully connected layer of the form
        Activation(Wa + b)

        Here:
        - a is of dimension (n,1), when trained with batches, it will be (n, m)
        - b is of dimension (n,1)
        - W is of dimension (r,n)
        - Activation is a function from R^r to R^r
    """
    def __init__(self, input_size, units, activation = activations.identity, optimizer = None):
        """
        input_size: the dimension of the input
        units: the dimension of the output
        activation: the activation function to be used, must be a class in activations.py, the default is identity
        optimizer: the optimizers to be used, when set to None it uses GD. Other optimizers are in optimizers.py
        """
        self.weights = np.random.randn(units, input_size) * np.sqrt(1.0 / input_size)
        self.biases = np.random.randn(units).reshape((units, 1))
        self.activation = activation()
        if optimizer is None:
            self.optimizer = None
        else:
            self.optimizer = optimizer()
        
    def forward(self, layer_input):
        self.input = layer_input.reshape((-1, layer_input.shape[-1])) #reshaping in case
        self.output = self.activation.forward(self.weights @ self.input + self.biases)
        
    def backward(self, d_output, learning_rate = 0.1):
        if isinstance(self.activation, activations.Softmax):
            dz = self.activation.backward(d_output)
        else:
            dz = (d_output.T * self.activation.backward()).T
            
        d_output = dz @ self.weights
        
        grad_W = dz.T @ self.input.T 
        grad_b = dz.sum(axis=0, keepdims=True).reshape(-1,1)
        
        if self.optimizer is None:
            self.weights = self.weights - learning_rate*grad_W
            self.biases = self.biases - learning_rate*grad_b
        else:
            self.optimizer.update(grad_W, grad_b)
            self.weights = self.weights - learning_rate*self.optimizer.param_W
            self.biases = self.biases - learning_rate*self.optimizer.param_b
            
        return d_output
        
