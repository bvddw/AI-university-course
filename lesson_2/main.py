import math


def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))


class neural_network:
    def __init__(self, input_data: list, number_of_levels: int, matrices_data: list) -> None:
        self.data = input_data  # data for input to the neurons
        self.levels = number_of_levels  # numbers of levels, including first (just input)
        self.matrices = matrices_data  # matrices for every level

    def learning(self):  # here we are starting learning
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
        for amount_of_steps in range(self.levels - 1):
            new_data = []
            cur_matrix = self.matrices[amount_of_steps]
            for vertex in range(len(cur_matrix)):
                cur_output = 0
                for index in range(len(self.data)):
                    cur_output += self.data[index] * cur_matrix[vertex][index]
                new_data.append(sigmoid(cur_output))
            self.data = new_data
        return self.data


data_to_input = [0.4, 0.2, 0.6, 0.5]
number_of_levels = 4
matrix1 = [
    [0.2, 0.6, 0.5, 0.3],
    [0.1, 0.8, 0.7, 0.2],
    [0.5, 0.3, 0.9, 0.7],
    [0.4, 0.6, 0.1, 0.8],
    [0.7, 0.9, 0.5, 0.6],
    [0.6, 0.3, 0.7, 0.2],
]
matrix2 = [
    [0.6, 0.3, 0.7, 0.4, 0.2, 0.5],
    [0.7, 0.5, 0.8, 0.1, 0.9, 0.4],
    [0.3, 0.6, 0.2, 0.7, 0.8, 0.3],
    [0.4, 0.9, 0.5, 0.2, 0.1, 0.7],
    [0.5, 0.7, 0.8, 0.6, 0.5, 0.2],
]
matrix3 = [
    [0.7, 0.6, 0.1, 0.5, 0.4],
    [0.8, 0.2, 0.3, 0.9, 0.1],
    [0.4, 0.9, 0.7, 0.5, 0.2],
]
newAI = neural_network(data_to_input, number_of_levels, [matrix1, matrix2, matrix3])
print(f"Data before:\n{newAI.data}")
print(f"Data after:\n{newAI.learning()}")
