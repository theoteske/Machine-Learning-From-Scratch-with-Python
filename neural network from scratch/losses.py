### Losses
import numpy as np
class MSE:
    def loss(self, predictions, y):
        l = np.mean((y - predictions)**2)
        return l
    def loss_gradient(self, predictions, y):
        grad = -2*(y - predictions)/y.shape[1]
        return grad.T
    
class CrossEntropy:
    
    def loss(self, predictions, y):
        #cross entropy loss l, add 1e-6 to the argument of the logarithm for numercal stability
        l = -np.sum(y*np.log(predictions+1e-6))/y.shape[1]
        return l.T
    
    def loss_gradient(self, predictions, y):
        #gradient of los
        grad = (-1/y.shape[1]) * (y / (predictions + 1e-6))
        return grad.T
