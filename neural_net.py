from layers import Dense, relu, sigmoid
from loss import *

from numpy import mean

class NeuralNet:
    def __init__(self, **hyperparameters):
        self.inputs_len = hyperparameters['inputs_len']
        self.hidden_len = hyperparameters['hidden_len']
        self.output_len = hyperparameters['output_len']

        self.network = [
            Dense(
                inputs_len = self.inputs_len, 
                outputs_len = self.hidden_len,
                learning_rate = hyperparameters['learning_rate'][0],
                activation = None
            ),
            relu(),
            Dense(
                inputs_len = self.hidden_len, 
                outputs_len = self.output_len,
                learning_rate = hyperparameters['learning_rate'][1],
                activation = None
            )
        ]

    def feedforward(self, image):
        memory = [image]

        inputs = image
        for layer in self.network:
            inputs = layer.forward(inputs)
            memory.append(inputs)
        return memory

    def backprop(self, logits, y_batch, neural_net_memory):
        """ Back propagate igennem nettet """
        # Beregn loss
        loss = softmax_crossentropy_with_logits(logits, y_batch)

        # Beregn første gradient
        loss_gradient = grad_softmax_crossentropy_with_logits(logits, y_batch)

        for layer_index in range(len(self.network))[::-1]:
            layer = self.network[layer_index]
            loss_gradient = layer.backward(
                neural_net_memory[layer_index],
                loss_gradient
            )

        return mean(loss)

    def predict(self, val_inputs):
        logits = self.feedforward(val_inputs)[-1]
        return logits.argmax(axis=-1)