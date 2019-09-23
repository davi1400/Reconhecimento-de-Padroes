from sklearn.model_selection import train_test_split
from src.Utils.utils import get_inputs, get_outputs, string_to_number_class, get_data, heaveside, normalize, \
    get_accuracy, get_confusion_matrix
from numpy.random import rand, permutation
from numpy import where, append, ones, array, zeros


class perceptron:
    def __init__(self, data=None, learning_rate=0.1, epochs=100, test_rate=0.2):
        if data is not None:
            self.number_lines = data.shape[0]
            self.number_columns = data.shape[1]
            self.X = get_inputs(data, self.number_columns - 1)
            self.classes = get_outputs(data, self.number_columns - 1)
            self.Y = string_to_number_class(self.classes)


        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_rate = test_rate


    def add_bias(self):
        self.X = append(-1 * ones((self.X.shape[0], 1)), self.X, 1)

    def train(self, batch=True):

        X_train, X_test, Y_train, Y_test = train_test_split(self.X, self.Y, test_size=self.test_rate)
        weights = rand(X_train.shape[1], 1)
        if batch:
            for epoch in range(self.epochs):
                # Hidden = X_train.dot(weights)
                Y_output = self.forward(X_train, weights)
                Error = Y_train - Y_output
                if abs(Error).sum() == 0:
                    print(epoch)
                    break

                self.backward(weights, Error, X_train)

        else:
            k = 0
            r = permutation(X_train.shape[0])  # Função equivalente ao randperm() do matlab
            for epoch in range(self.epochs):
                # Hidden = X_train.dot(weights)
                Y_output = self.forward(X_train[r[k]], weights)
                Error = Y_train[r[k]] - Y_output

                if abs((Y_train - heaveside(X_train.dot(weights))).sum()) == 0:
                    print(epoch)
                    break

                self.backward(weights, Error, X_train[r[k]], online=True)

                k += 1
                # Verificar se já passou por todos os exemplos se sim , fazer novamente randperm() e colocar o contador no 0
                if k >= r.shape[0]:
                    r = permutation(X_train.shape[0])
                    k = 0

        return weights, X_test, Y_test

    def test(self, weights, X_test, Y_test, confusion_matrix=False):
        Y_output = self.forward(X_test, weights)
        accuracy = get_accuracy(Y_output, Y_test)
        if confusion_matrix:
            return accuracy, get_confusion_matrix(Y_output, Y_test)
        else:
            return accuracy

    def forward(self, X, weights):
        return heaveside(X.dot(weights))

    def backward(self, weights, Error, X, online=False):
        if online:
            weights += array((self.learning_rate * Error[0] * X), ndmin=2).T
        else:
            weights += (self.learning_rate * Error.T.dot(X)).T
        return weights

    def transform_binary(self, name):
        self.Y[where(self.classes == name)] = 0
        self.Y[where(self.classes != name)] = 1







