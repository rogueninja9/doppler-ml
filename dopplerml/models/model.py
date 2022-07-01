from abc import ABC, abstractmethod
from ..errors.error import InputError


class Model(ABC):
    def __init__(self):
        self.X = None
        self.y = None
        
    def prepareInputs(self,X,y):
        if len(X.shape)!=2:
            raise InputError('Expected X of shape (m,n) but got shape '+str(X.shape))
        if len(y.shape)!=2:
            raise InputError('Expected y of shape (m,1) but got shape '+str(y.shape))
        if X.shape[0]!=y.shape[0]:
            raise InputError('Expected X of shape (m,n) and y of shape (m,1). But got X shape '+str(X.shape)+' and y shape '+str(y.shape))
        self.X=X
        self.y=y
    
    @abstractmethod
    def fit(self):
        pass
    
    @abstractmethod
    def predict(self):
        pass