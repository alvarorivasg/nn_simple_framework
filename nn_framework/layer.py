import numpy as np

class Dense:
    def __init__(self, m_inputs, n_outputs):
        self.m_inputs = int(m_inputs)
        self.n_outputs = int(n_outputs)
        self.learning_rate=.05
        #randomly initializing weights between -1 and 1
        #+1 on m_inputs in order to account for the bias node
        self.weights = np.random.sample(size = (self.m_inputs + 1, self.n_outputs))*2 -1
        #to avoid errors if we reference before the first iteration:
        self.x = np.zeros((1, m_inputs + 1))
        self.y = np.zeros((1, m_inputs))

    def forward_prop(self, inputs):
        bias = np.ones((1, 1))
        self.x = np.concatenate((inputs, bias), axis = 1)
        self.y = self.x @ self.weights
        return self.y
    

        