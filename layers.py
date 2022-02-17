import numpy as np
from numpy import dot, exp, maximum, sqrt
from numpy.random import uniform, normal

class Dense:
    
    def __init__(self, inputs_len, outputs_len, learning_rate=0.1, activation=None):
        self.inputs_len = inputs_len
        self.outputs_len = outputs_len

        self.learning_rate = learning_rate

        self.weights = normal(
            loc = 0.0,
            scale = sqrt(2/(inputs_len+outputs_len)),
            size = (inputs_len, outputs_len)
        )
        self.biases = uniform(-1,1, outputs_len)
        
        self.activate_fn = activation

    def forward(self, x):
        # Lav et pass forward beregning
        outputs = dot(x, self.weights) + self.biases

        if self.activate_fn == None:
            return outputs
        return self.activate_fn.forward(outputs)

    def backward(self, inputs, grad_outputs):
        """ Beregn df/dx = df/ddense * ddense / dx """

        ###
        # Beregn differentialet af activation_function her?
        ###

        grad_inputs = np.dot(grad_outputs, self.weights.T)

        grad_weights = dot(inputs.T, grad_outputs)
        grad_biases = grad_outputs.mean(axis=0)*inputs.shape[0]

        assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
    
        # Vi kan nu lave et stokastisk gradient decent skridt
        self.weights = self.weights - self.learning_rate * grad_weights
        self.biases = self.biases - self.learning_rate * grad_biases


        return grad_inputs

class relu:

    def __init__(self): pass

    def forward(self, x):
        "Return alle værdier der er over 0, ellers returner 0"
        return maximum(0,x)

    def backward(self, x, grad_output):
        relu_grad = x > 0
        return relu_grad * grad_output

class sigmoid:

    def __init__(self): pass

    def forward(self, x):
        "Normalisering af alle værdierne mellem 1 og 0"
        return 1 / (1 + exp(-x))

    def backward(self, x, grad_output):
        forward = self.forward(x)
        return forward - forward * forward
