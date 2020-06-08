import numpy as np

class TanH:
    '''This is not well designed. calc_d() should be using the previous result of calc() and not tanh(v),
    Go check what is passed to calc_d() in layer.py (it is the output of the layer, and not the input to the activation function)
    to see what I mean. Let's keep it like this until the end and then see what I can do'''
    @staticmethod
    def calc(v):
        return np.tanh(v)

    @staticmethod
    def calc_d(v):
        return 1 - np.tanh(v) ** 2

'''class TanH:
    This would be the way to properly calculate the derivative. But it is not valid with our framework
    def __init__(self):
        # Including this class attribute lets TanH re-use its results.
        self.calc_fwd = None

    def calc(self, v):
        self.calc_fwd = np.tanh(v)
        return self.calc_fwd

    def calc_d(self, v):
        return 1 - self.calc_fwd ** 2'''

 

class ReLU:
    '''Same mistake as with TanH. calc_d should do the maths with the result from calc, not with v'''
    @staticmethod
    def calc(v):
        return np.maximum(0, v)

    @staticmethod
    def calc_d(v):
        derivative = np.zeros(v.shape)
        derivative[np.where(v > 0)] = 1
        return derivative
