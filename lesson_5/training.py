from main import NeuralNetwork
import numpy as np


def train_fcn(network):
    training_file = open("mnist_train.csv", 'r')
    training_list = training_file.readlines()
    training_file.close()
    epochs = 5
    print("neural network training...")
    for e in range(epochs):
        print("epoch", e + 1)
        for record in training_list:
            all_values = record.split(',')
            inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
            targets = np.zeros(10) + 0.01
            targets[int(all_values[0])] = 0.99
            network.train(inputs, targets)


def test_fcn(network):
    test_file = open("mnist_test.csv", 'r')
    test_list = test_file.readlines()
    test_file.close()
    scorecard = []
    print("neural network testing...")
    for record in test_list:
        all_values = record.split(',')
        correct_label = int(all_values[0])
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        outputs = network.query(inputs)
        label = np.argmax(outputs)
        if label == correct_label:
            scorecard.append(1)
        else:
            scorecard.append(0)
    scorecard_array = np.asarray(scorecard)
    print("Efficiency:", scorecard_array.sum() / scorecard_array.size * 100, "%")


if __name__ == "__main__":
    n = NeuralNetwork()
    train_fcn(n)
    test_fcn(n)
    n.save_weights()
