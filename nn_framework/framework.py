import numpy as np

class ANN:
    def __init__(self, model=None, expected_range=(-1,1)):
        self.layers = model
        self.n_iter_train=int(1e8)
        self.n_iter_eval=int(1e6)
        self.expected_range=expected_range


    def train(self, training_set):
        for iter in range(self.n_iter_train):
            x=self.normalize(next(training_set()).ravel())
            print(x)

    def evaluate(self, evaluation_set):
        for iter in range(self.n_iter_eval):
            x=self.normalize(next(evaluation_set()).ravel())

    def normalize(self, values):
        '''Transforms input-output values so they fall between -0.5 and 0.5'''
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = max_val - min_val
        offset_factor = min_val
        return (values - offset_factor) / scale_factor - .5

    def denormalize(self,transformed_values):
        min_val = self.expected_range[0]
        max_val = self.expected_range[1]
        scale_factor = max_val - min_val
        offset_factor = min_val
        return (transformed_values + .5) * scale_factor + offset_factor


