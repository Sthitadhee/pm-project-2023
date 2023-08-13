import numpy as np

def inv_logit(x):
    return np.exp(x)/(1 + np.exp(x))
