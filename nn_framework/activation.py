import numpy as np

class TanH:
    '''This is not well designed. calc_d() should be using the previous result of calc() and not tanh(v),
    Go check what is passed to calc_d() in layer.py (it is the output of the layer, and not the input to the act function)
    to see what I mean. Let's keep it like this until the end and then see what I can do'''
    @staticmethod
    def calc(v):
        return np.tanh(v)

    @staticmethod
    def calc_d(v):
        return 1 - np.tanh(v) ** 2
 

class ReLU:
    '''I'm pretty sure this is fine'''
    @staticmethod
    def calc(v):
        return np.maximum(0, v)

    @staticmethod
    def calc_d(v):
        derivative = np.zeros(v.shape)
        derivative[np.where(v > 0)] = 1
        return derivative
