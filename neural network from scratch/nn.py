import numpy as np
from scipy.special import softmax
import losses

class Sequential:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, inputs):
        for layer in self.layers:
            layer.forward(inputs)
            inputs = layer.output
        return inputs

    def backward(self, d_output, learning_rate):
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output, learning_rate)

    def train(self, X, y, epochs, learning_rate, batch_size, loss = losses.MSE):

        N = X.shape[1]
        n_batches = N // batch_size
        
        
        for epoch in range(epochs):
            indices = np.random.permutation(N)
            X_shuffle = X[:,indices]
            y_shuffle = y[:,indices]
            
            for i in range(n_batches):
                X_batch = X[:,i * batch_size : (i+1) * batch_size]
                y_batch = y[:,i * batch_size : (i+1) * batch_size]
                predictions = self.forward(X_batch)

                d_loss = loss().loss_gradient(predictions, y_batch)
                self.backward(d_loss, learning_rate)
                
            predictions = self.forward(X)
            l = loss().loss(predictions, y)
            print(f"Epoch {epoch}, Loss: {l}")

    def predict(self, X):
        return self.forward(X)