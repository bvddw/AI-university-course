import numpy as np


class NeuralNetwork:
    def __init__(self, inputnodes=784, hidennodes=200, outputnodes=10, learningrate=0.1):
        self.__lr = learningrate
        self.__wih = np.random.normal(0.0, pow(inputnodes, -0.5), (hidennodes, inputnodes))
        self.__who = np.random.normal(0.0, pow(hidennodes, -0.5), (outputnodes, hidennodes))

    def query(self, inputs_list):
        inputs = np.array(inputs_list, ndmin=2).T
        hidden_inputs = np.dot(self.__wih, inputs)
        hidden_outputs = self.__activation_function(hidden_inputs)
        final_inputs = np.dot(self.__who, hidden_outputs)
        outputs = self.__activation_function(final_inputs)
        return outputs

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.__wih, inputs)
        hidden_outputs = self.__activation_function(hidden_inputs)
        final_inputs = np.dot(self.__who, hidden_outputs)
        final_outputs = self.__activation_function(final_inputs)
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.__who.T, output_errors)
        self.__who += self.__lr * np.dot((output_errors * final_outputs * (1 - final_outputs)), hidden_outputs.T)
        self.__wih += self.__lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)), inputs.T)

    def save_weights(self):
        np.save("weights_input_hidden.npy", self.__wih)
        np.save("weights_hidden_output.npy", self.__who)

    def load_weights(self, path_ih, path_ho):
        w_ih = np.load(path_ih)
        self.__wih = w_ih
        w_ho = np.load(path_ho)
        self.__who = w_ho

    def __activation_function(self, inputSignal):
        return 1 / (1 + np.exp(-inputSignal))
