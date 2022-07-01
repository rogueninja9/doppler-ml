from .model import Model
from ..optimizers import GradientDescent
from ..utils import addConstant
import numpy as np

class LinearRegression(Model):
    def __init__(self, optimizer = 'gradient_descent'):
        if optimizer=='gradient_descent':
            self.optimizer = GradientDescent()
        super().__init__()
        
    def fit(self,X,y):
        self.prepareInputs(X,y)
        self.coeff_ = np.zeros(X.shape[1]+1)
        self.coeff_ = self.optimizer.optimize(self.coeff_, addConstant(X), y)
        
    def predict(self,X):
        return np.dot(addConstant(X), self.coeff_).reshape(X.shape[0],1)