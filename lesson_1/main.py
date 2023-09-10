class LinearClassifier:
    def __init__(self, weight, learning_rate):
        self.weight = weight  # parameter to regulate
        self.learning_rate = learning_rate  # coefficient of learning speed

    def query(self, input):  # regulating parameter
        return self.weight * input

    def train(self, input, target):  # training
        y = self.query(input)
        e = target - y
        delta_w = self.learning_rate * e / input
        self.weight += delta_w


# bugs data
width = [2, 9, 3, 10, 4, 11, 6, 12, 3, 11, 4, 12, 5, 12]
length = [12, 2, 10, 3, 11, 4, 12, 5, 8, 2, 9, 2, 10, 3]
# zip in one list of tuples (width, length)
data_to_check = [(width[i], length[i]) for i in range(14)]

# initiating classifier with weight = 0.05, learning_rate = 0.1
classifier = LinearClassifier(0.05, 0.1)

# classifying before training
print("BEFORE TRAINING")
number_of_example = 1
for input, target in data_to_check:
    if input * classifier.weight < target:
        print(f'Example {number_of_example}: caterpillar')
    else:
        print(f'Example {number_of_example}: ladybug')
    number_of_example += 1

# changing in training process
print("TRAINING PROCESS")
for input, target in data_to_check:
    classifier.train(input, target)
    print(f"New regulated parameter: {classifier.weight}")

# classifying after training process
print("AFTER TRAINING")
number_of_example = 1
for input, target in data_to_check:
    if input * classifier.weight < target:
        print(f'Example {number_of_example}: caterpillar')
    else:
        print(f'Example {number_of_example}: ladybug')
    number_of_example += 1
