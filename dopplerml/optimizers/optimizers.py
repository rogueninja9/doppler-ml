from .optimizer import Optimizer
import numpy as np

class GradientDescent(Optimizer):
    def __init__(self,learning_rate = 1e-8, epochs = 1e5):
        self.learning_rate = learning_rate
        self.epochs = epochs
    def optimize(self,theta,X,y):
        m = X.shape[0]
        predict = lambda x,t: np.dot(x,t).reshape(X.shape[0],1)
        cost = lambda x,y,t: 1/(2*m)*np.sum(np.power((predict(x,t) - y),2))
        for _ in range(int(self.epochs)):
            theta_new = theta - self.learning_rate * np.sum((predict(X,theta).reshape((X.shape[0]),1) - y)*X , axis=0)
            theta = theta_new
        return theta