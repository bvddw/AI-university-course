import math


def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))


class NeuralNetwork:
    def __init__(self, input_data: list, number_of_levels: int, matrices_data: list) -> None:
        self.input_data = input_data
        self.data = input_data  # data for input to the neurons
        self.levels = number_of_levels  # numbers of levels, including first (just input)
        self.matrices = matrices_data  # matrices for every level
        self.output_data = [self.data]
        self.mistakes = []
        self.learning_rate =  10
        self.delta_mistakes = []

    def count_output(self):  # here we are starting learning
        """
        In this function, we go through each level according to the algorithm:
            1) We take the list of data, which is responsible for the values in the vertices of the current level.
            2) Using the matrix of coefficients for each pair of vertices (one from our level, the other from the next),
                we get new data for the vertices of the next level, which we will validate.
                For each element of new data, we use a sigmoid, and then we overwrite the data list in self.data.
            3) We continue to do these operations until we get the result from the last level == output.
        :return:
            The list of data for vertices that passes through each neuron.
        """
        self.data = self.input_data
        for amount_of_steps in range(self.levels - 1):
            new_data = []
            cur_matrix = self.matrices[amount_of_steps]
            for vertex in range(len(cur_matrix)):
                cur_output = 0
                for index in range(len(self.data)):
                    cur_output += self.data[index] * cur_matrix[vertex][index]
                new_data.append(sigmoid(cur_output))
            self.output_data.append(new_data)
            self.data = new_data
        return self.data

    def mistakes_counter(self, real_outputs: tuple):
        """
        counting mistakes on each vertex of our neural network
        :param real_outputs: tuple of correct results
        :return: None
        """
        e_outputs = [real_outputs[i] - self.data[i] for i in range(len(self.data))]
        self.mistakes.append(e_outputs)
        level = self.levels - 2
        while level >= 0:  # going from output level to input level
            current_Wmatrix = self.matrices[level]  # taking current matrix of coefficients
            new_e_outputs = []
            for i in range(len(current_Wmatrix[0])):  # calculating vector of node errors, using
                mistake = 0
                for j in range(len(current_Wmatrix)):
                    mistake = mistake + e_outputs[j] * current_Wmatrix[j][i]
                new_e_outputs.append(mistake)
            e_outputs = new_e_outputs
            self.mistakes.append(e_outputs)  # adding mistakes for current level to matrix of mistakes
            level -= 1

    def deltas_counter(self):
        """
        counting mistakes of weight coefficients and updating them
        :return:
        """
        all_errors = self.mistakes[::-1]
        for i in range(len(self.matrices)):
            cur_weights = self.matrices[i]
            cur_errors = all_errors[i + 1]
            o_vector = self.output_data[i]
            errors_vector = []
            for j in range(len(self.output_data[i + 1])):
                sigm_arg = 0
                cur_weights_row = cur_weights[j]
                for k in range(len(self.output_data[i])):
                    sigm_arg += self.output_data[i][k] * cur_weights_row[k]
                sigm_result = sigmoid(sigm_arg) * (1 - sigmoid(sigm_arg))
                errors_vector.append(cur_errors[j] * sigm_result)
            cur_deltas_matrix = []
            for p in range(len(errors_vector)):
                row = []
                for q in range(len(o_vector)):
                    row.append(self.learning_rate * errors_vector[p] * o_vector[i])
                cur_deltas_matrix.append(row)
            self.delta_mistakes.append(cur_deltas_matrix)
            for p in range(len(errors_vector)):
                for q in range(len(o_vector)):
                    cur_weights[p][q] = cur_weights[p][q] + cur_deltas_matrix[p][q]

    @staticmethod
    def training(amount_of_repeating, correct_values=(0.1, 0.9, 0.5)):
        """
        We call the methods sequentially:
        1) calculate the output (count_output)
        2) count errors by level (mistakes_counter)
        3) update the weighting coefficients (deltas_counter)

        :param amount_of_repeating: take from user number of repeating learn processes
        :param correct_values: just correct values, with which we are comparing our output
        :return: None
        """
        print(f"Data before training:\n{newAI.data}")
        for _ in range(amount_of_repeating):
            newAI.count_output()
            newAI.mistakes_counter(correct_values)
            newAI.deltas_counter()
        print(f"Data after training:\n{newAI.data}")


if __name__ == "__main__":
    # data for neural network
    data_to_input = [0.6, 0.2, 0.7, 0.4]
    number_of_levels = 4
    Wmatrix1 = [
        [0.3, 0.8, 0.7, 0.2],
        [0.6, 0.9, 0.1, 0.4],
        [0.4, 0.5, 0.6, 0.9],
        [0.8, 0.2, 0.3, 0.7],
        [0.5, 0.7, 0.8, 0.1],
    ]
    Wmatrix2 = [
        [0.7, 0.3, 0.5, 0.4, 0.9],
        [0.5, 0.2, 0.8, 0.3, 0.1],
        [0.8, 0.6, 0.3, 0.5, 0.2],
        [0.3, 0.7, 0.4, 0.1, 0.9],
        [0.4, 0.8, 0.6, 0.2, 0.7],
        [0.9, 0.4, 0.2, 0.6, 0.1],
    ]
    Wmatrix3 = [
        [0.5, 0.2, 0.7, 0.1, 0.8, 0.4],
        [0.7, 0.3, 0.8, 0.9, 0.1, 0.6],
        [0.4, 0.5, 0.2, 0.7, 0.9, 0.3],
    ]
    real_values = (0.1, 0.9, 0.5)
    # network initialization
    newAI = NeuralNetwork(data_to_input, number_of_levels, [Wmatrix1, Wmatrix2, Wmatrix3])
    newAI.count_output()
    # learning process
    newAI.training(1000)
    """
    :output
        Data before training:
        [0.9099615852368447, 0.947246945399052, 0.9279198423559678]
        Data after training:
        [0.10000000000000003, 0.8999999999999999, 0.5]
    """
