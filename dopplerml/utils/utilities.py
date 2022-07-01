import numpy as np


addConstant = lambda x: np.c_[x,np.ones(x.shape[0]).reshape(x.shape[0],1)]