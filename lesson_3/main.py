import math


def sigmoid(x):
    return 1 / (1 + math.pow(math.e, -x))


class neural_network:
    def __init__(self, output_data: list, number_of_levels: int, matrices_data: list, real_output_data: list) -> None:
        self.output_data = output_data  # data for input to the neurons
        self.levels = number_of_levels  # numbers of levels, including first (just input)
        self.matrices = matrices_data  # matrices for every level
        self.real_output_data = real_output_data  # vector of real output
        self.mistakes = []  # matrix of mistakes for each level

    def mistakes_counter(self):
        """
        In this method, we use backpropagation and weighting matrices to calculate the error at each node.
        """
        e_outputs = [self.real_output_data[i] - self.output_data[i] for i in range(len(self.output_data))]
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


output_data = [0.2, 0.5, 0.3]
number_of_levels = 4
Wmatrix1 = [
    [0.3, 0.7, 0.4, 0.2, 0.1, 0.8],
    [0.9, 0.1, 0.6, 0.5, 0.2, 0.7],
    [0.2, 0.3, 0.8, 0.4, 0.3, 0.5],
    [0.7, 0.6, 0.9, 0.1, 0.8, 0.6],
    [0.1, 0.5, 0.4, 0.3, 0.7, 0.5],
]
Wmatrix2 = [
    [0.5, 0.3, 0.2, 0.7, 0.9],
    [0.7, 0.1, 0.8, 0.2, 0.6],
    [0.3, 0.8, 0.7, 0.1, 0.5],
    [0.9, 0.2, 0.6, 0.7, 0.3],
]
Wmatrix3 = [
    [0.4, 0.1, 0.6, 0.7],
    [0.3, 0.5, 0.8, 0.9],
    [0.6, 0.4, 0.1, 0.2],
]
real_values = [0.6, 0.2, 0.4]
newAI = neural_network(output_data, number_of_levels, [Wmatrix1, Wmatrix2, Wmatrix3], real_values)
newAI.mistakes_counter()
for level, mistake_vector in enumerate(newAI.mistakes[::-1]):
    print(f"Vector of mistakes on level {level + 1} is: {mistake_vector}")