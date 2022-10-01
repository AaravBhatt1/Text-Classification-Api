import numpy as np


class Dense:
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim  # nummber of neurons feeded in to layer
        self.output_dim = output_dim  # number of neurons in layer
        self.weights = np.random.randn(input_dim, output_dim)
        self.biases = np.random.randn(1, output_dim)
        self.wmomentum = np.zeros_like(self.weights)
        self.bmomentum = np.zeros_like(self.biases)

    def forward(self, inputs):
        assert inputs.shape[1] == self.input_dim
        # remembers the inputs for backpropagation
        self.inputs = inputs
        # multiplies inputs by the weights and adds biases
        self.outputs = inputs.dot(self.weights) + self.biases
        return self.outputs

    def backward(self, dvalues):
        assert dvalues.shape[1] == self.output_dim
        # calculates the derivatives of output in respect of weights and multiplies by corresponding dvalue
        self.dweights = self.inputs.T.dot(dvalues)
        # the derivative of adding the biases is only 1, so we use 1 * dvalues (and sum it)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # calculates the gradient on the inputs as we did with weights
        self.dinputs = dvalues.dot(self.weights.T)
        return self.dinputs


class Relu:
    def forward(self, inputs):
        # remembers the inputs for backpropagetion
        self.inputs = inputs
        # use the max function to check if a value is less than 0 and then make it 0 if it is
        self.outputs = np.maximum(0, inputs)
        return self.outputs

    def backward(self, dvalues):
        # it is 1 * dvalue when the input was more then 0 otherwise it is 0
        self.dinputs = dvalues
        self.dinputs[self.inputs < 0] = 0
        return self.dinputs


class Softmax:
    def forward(self, inputs):
        # get unnormalised probabilites by exponetiating and comparing to the maximum
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalise the probabilities by dividing by the total
        outputs = exp_values / \
            np.sum(exp_values, axis=1, keepdims=True)
        return outputs


# this is because it is easier to backpropate both softmax and cross-entropy loss at once
class SoftmaxWithCategoricalCrossEntropy:
    def forward(self, inputs, true_outputs):
        # remembers the inputs and outputs
        self.inputs = inputs
        self.true_outputs = true_outputs
        self.softmax_outputs = Softmax().forward(inputs)
        # find the batch_size
        self.batch_size = len(inputs)
        # clip data to prevent division by 0 (both sides to prevent bias)
        clipped_softmax_outputs = np.clip(self.softmax_outputs, 1e-7, 1 - 1e-7)
        # multiply incorrect values by 0
        correct_values = np.sum(clipped_softmax_outputs * true_outputs, axis=1)
        self.loss = -np.log(correct_values)
        return self.loss

    def backward(self):
        true_values_disc = np.argmax(self.true_outputs, axis=1)
        self.dinputs = self.softmax_outputs.copy()
        self.dinputs[range(self.batch_size), true_values_disc] -= 1
        # normalise the dinputs
        self.dinputs /= self.batch_size
        return self.dinputs


class SGD:
    def __init__(self, learning_rate=1, decay=0, momentum=0.):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.original_learning_rate = learning_rate
        self.momentum = momentum

    def update_decay(self):
        if self.decay:
            self.learning_rate = self.original_learning_rate * \
                (1 / (1 + self.decay * self.iterations))
        self.iterations += 1

    def update_layer(self, layer):
        layer.wmomentum = self.momentum * layer.wmomentum - \
            self.learning_rate * layer.dweights
        layer.bmomentum = self.momentum * layer.bmomentum - \
            self.learning_rate * layer .dbiases
        layer.weights += layer.wmomentum
        layer.biases += layer.bmomentum
