from sklearn.model_selection import train_test_split
from src.Utils.utils import get_inputs, get_outputs, string_to_number_class, get_data, heaveside, normalize
from numpy.random import rand


class perceptron:
    def __init__(self, data, learning_rate, epochs, test_rate=0.2):
        self.number_lines = data.shape[0]
        self.number_columns = data.shape[1]
        self.X = get_inputs(data, self.number_columns - 1)
        self.classes = get_outputs(data, self.number_columns - 1)
        self.Y = string_to_number_class(self.classes)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_rate = test_rate

    def train(self, batch=True, nomalizer=False):
        if nomalizer:
            self.X = normalize(self.X)
        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.test_rate)
        weights = rand(X_train.shape[1], 1)
        if batch:
            for epoch in range(self.epochs):
                # Hidden = X_train.dot(weights)
                Y_output = self.forward(X_train, weights)
                Error = Y_train - Y_output
                print(Error)
                if abs(Error).sum() == 0:
                    print(epoch)
                    break
                # weights += (self.learning_rate * Error.T.dot(X_train)).T
                weights += self.backward(weights, Error, X_train)

        return weights, X_test, Y_test

    def test(self, weights, X_test, Y_test, confusion_matrix=False):
        return 0

    def forward(self, X, weights):
        Hidden = X.dot(weights)
        Y_output = heaveside(Hidden)
        return Y_output

    def backward(self, weights, Error, X):
        weights += (self.learning_rate * Error.T.dot(X)).T
        return weights


if __name__ == '__main__':
    data = get_data("OR", type='csv')
    p = perceptron(data, 0.015, 100)
    accuracys = []
    confusions_matrix = []
    for realization in range(20):
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)