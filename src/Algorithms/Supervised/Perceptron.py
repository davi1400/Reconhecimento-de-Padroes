import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.Utils.utils import get_inputs, get_outputs, string_to_number_class, get_data, heaveside, normalize, \
    get_accuracy, get_confusion_matrix, sigmoid, get_error_rate
from numpy.random import rand, permutation
from numpy import where, append, ones, array, zeros
from src.Utils.ColorMap import ColorMap
from matplotlib.colors import ListedColormap
from src.Utils.CrossValidate import CrossValidation


class perceptron:
    def __init__(self, data=None, learning_rate=0.1, epochs=100, test_rate=0.2, validate=None, type=None, logist=False):
        if data is not None:
            self.number_lines = data.shape[0]
            self.number_columns = data.shape[1]
            self.X = get_inputs(data, self.number_columns - 1)
            self.classes = get_outputs(data, self.number_columns - 1)
            if isinstance(self.classes[0], str):
                self.Y = string_to_number_class(self.classes)
            else:
                self.Y = self.classes

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.test_rate = test_rate
        self.percents = []
        self.stds = []
        self.variances = []
        self.validate = validate
        self.type = type
        self.logistic = logist
        if type == 1:
            self.g = heaveside
        elif type == 2:
            self.g = sigmoid

    def variance(self):
        self.variances.append(np.var(self.percents))

    def standard_deviation(self):
        self.stds.append(np.std(self.percents))

    def add_bias(self):
        self.X = append(-1 * ones((self.X.shape[0], 1)), self.X, 1)

    # def validate(self):
    #     if self.validate:
    #         grid = [0.01, 0.015]
    #         validation = CrossValidation(X_train, Y_train, grid, self)
    #         validation.validate()

    def split(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y,
                                                                                test_size=0.01)

    def train(self, batch=True, normalizer=True):

        weights = zeros((self.X_train.shape[1], 1))
        if batch:
            for epoch in range(self.epochs):
                # Hidden = X_train.dot(weights)
                H_output = self.forward(self.X_train, weights)
                Y_output = self.predict(H_output)
                Error = array(self.Y_train, ndmin=2).T - Y_output

                accuracy = self.test(weights, self.X_test, self.Y_test, confusion_matrix=True)
                self.percents.append(accuracy)

                if abs(Error).sum() == 0:
                    # print(epoch)
                    break

                self.backward(weights, Error, self.X_train, self.learning_rate, H_output)

        else:
            k = 0
            r = permutation(self.X_train.shape[0])  # Função equivalente ao randperm() do matlab
            for epoch in range(self.epochs):
                # Hidden = X_train.dot(weights)
                Y_output = self.forward(self.X_train[r[k]], weights)
                if Y_output == 0:
                    Y_output = -1.0
                Error = self.Y_train[r[k]] - Y_output

                if abs((self.Y_train - heaveside(self.X_train.dot(weights))).sum()) == 0:
                    # print(epoch)
                    break
                if Error != 0.0:
                    self.backward(weights, Error, self.X_train[r[k]], self.learning_rate,  self.Y_train[r[k]],online=True)

                k += 1
                # Verificar se já passou por todos os exemplos se sim , fazer novamente randperm() e colocar o contador no 0
                if k >= r.shape[0]:
                    r = permutation(self.X_train.shape[0])
                    k = 0

        self.variance()
        self.standard_deviation()
        self.percents = []
        print("Erro dentro da amostra", self.test(weights, self.X_train, self.Y_train, inverse=True))
        return weights, self.X_test, self.Y_test

    def test(self, weights, X_test, Y_test, confusion_matrix=False, inverse=False):

        H_output = self.forward(X_test, weights)
        Y_output = self.predict(H_output)
        if inverse:
            error_rate = get_error_rate(Y_output, array(Y_test, ndmin=2).T)
            return error_rate
        else:
            accuracy = get_accuracy(Y_output, array(Y_test, ndmin=2).T)
            return accuracy

    def forward(self, X, weights):
        if self.type != 1:
            return self.g(self.logistic, X.dot(weights))
        else:
            return (X.dot(weights))

    def predict(self, y):
        y = y.copy()
        if self.logistic:
            for i in range(0, len(y)):
                if y[i][0] > 0.5:
                    y[i][0] = 1
                else:
                    y[i][0] = 0
        else:
            for i in range(0, len(y)):
                if y[i][0] >= 0:
                    y[i][0] = 1
                else:
                    y[i][0] = -1

        return y

    def backward(self, weights, Error, X, learning_rate, Y, online=False):
        if online:
            # weights += array((Y * X), ndmin=2).T
            weights += array((learning_rate * Error * X), ndmin=2).T
        else:
            if self.type == 1:
                # heaveside function
                derivate_y = 1
            elif self.type == 2:
                # sigmoid function
                if self.logistic:
                    # sigmoid logistica
                    derivate_y = self.get_derivate(Y)
                else:
                    # sigmoid tagente hiperbolica
                    derivate_y = self.get_derivate(Y)

            weights += (learning_rate * (derivate_y * Error).T.dot(X)).T
#             weights += Y*X

        return weights

    def transform_binary(self, name, flag=False):
        if flag:
            self.Y[where(self.classes == name)] = 1
            self.Y[where(self.classes != name)] = -1
        else:
            self.Y[where(self.classes == name)] = 1
            self.Y[where(self.classes != name)] = 0

    def get_derivate(self, *argv):
        if self.type == 1:
            return 1
        else:
            Y = argv[0]
            if self.logistic:
                # derivate of sigmoid logistic
                derivate = Y * (1 - Y)
                return derivate
            else:
                # derivate of sigmoid hyperbolic tangent
                derivate = 0.5 * (1 - Y ** 2)
                return derivate
