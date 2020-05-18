import numpy as np


def get_data_sets():
    '''Creates generator functions we will use for testing our neural network'''
    examples = [np.array([[0, 0], [0, 1]]), np.array([[0, 0], [1, 1]]), np.array([[0, 1], [1, 1]]),
                np.array([[1, 1], [1, 0]]), np.array([[1, 1], [0, 1]]), np.array([[1, 0], [1, 0]]),
                np.array([[1, 0], [0, 1]]), np.array([[0, 1], [0, 1]]), np.array([[1, 1], [0, 0]]),
                np.array([[0, 1], [1, 0]])]

    def training_set():
        while True:
            idx=np.random.choice(len(examples))
            yield examples[idx]

    def evaluation_set():
        while True:
            idx=np.random.choice(len(examples))
            yield examples[idx]

    return training_set, evaluation_set