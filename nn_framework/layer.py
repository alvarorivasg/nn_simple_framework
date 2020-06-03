import numpy as np

class Dense:
    def __init__(self, m_inputs, n_outputs, activate):
        self.m_inputs = int(m_inputs)
        self.n_outputs = int(n_outputs)
        self.activate = activate

        self.learning_rate=.001

        #randomly initializing weights between -1 and 1
        #+1 on m_inputs in order to account for the bias node
        self.weights = np.random.sample(size = (self.m_inputs + 1, self.n_outputs))* 2 - 1
        #to avoid errors if we reference before the first iteration:
        self.x = np.zeros((1, self.m_inputs + 1))
        self.y = np.zeros((1, self.m_inputs))

    def forward_prop(self, inputs):
        bias = np.ones((1, 1))
        self.x = np.concatenate((inputs, bias), axis = 1)
        v = self.x @ self.weights
        self.y = self.activate.calc(v)
        return self.y
    
    def back_prop(self, de_dy):
        dy_dv = self.activate.calc_d(self.y)
        dy_dw = self.x.T @ dy_dv #chain rule: the first element is dv_dw. Couldn't this expression be twisted and so the .T not needed?
        de_dw = de_dy * dy_dw
        self.weights -= de_dw * self.learning_rate
        de_dx = (dy_dv * de_dy) @ self.weights.T #chain rule, as weights is dv_dx
        return de_dx[:,:-1] #to avoid the weight associated with the bias node


        