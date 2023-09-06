class LinearClassifier:
    def __init__(self, weight, learning_rate):
        self.weight = weight  # parameter to regulate
        self.learning_rate = learning_rate  # coefficient of learning speed

    def query(self, input):  # take data from user
        return self.weight * input

    def train(self, input, target):  # training
        y = self.query(input)
        e = target - y
        delta_w = self.learning_rate * e / input
        self.weight += delta_w


input = [2, 9, 3, 10, 4, 11, 6, 12, 3, 11, 4, 12, 5, 12]
target = [12, 2, 10]