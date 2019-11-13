from numpy import zeros, concatenate, ones, where, array, argmin, log, exp
from numpy.random import rand
from src.Utils.utils import sigmoid


class LogisticRegression:
    def __init__(self, eta=0.1, ephocs=100):
        self.eta = eta
        self.ephocs = ephocs

    def train(self, x_train, y_train):
        bias = -1. * ones((x_train.shape[0]))
        x_train = concatenate((array(bias, ndmin=2).T, x_train), axis=1)
        N_features = x_train.shape[1]
        Wheigts = rand(N_features, 1)

        for epoch in range(self.ephocs):
            if array(y_train, ndmin=2).shape[1] != 1:
                aux_y = array(y_train, ndmin=2).T
            else:
                aux_y = array(y_train, ndmin=2)
            u = aux_y * (x_train.dot(Wheigts))
            H = sigmoid(True, -u)
            Y = self.predict(H)

            # Error = Y - array(y_train, ndmin=2).T
            Error = log(1 - exp(-1. * u))

            Wheigts += self.gradient_descent(x_train, y_train, u, Error)

        return Wheigts

    def test(self, Wheigts, x_test, y_test):
        bias = -1. * ones((x_test.shape[0]))
        x_test = concatenate((array(bias, ndmin=2).T, x_test), axis=1)
        u = x_test.dot(Wheigts)
        H = sigmoid(True, u)
        Y = self.predict(H)

        if array(y_test, ndmin=2).shape[1] != 1:
            aux_y = array(y_test, ndmin=2).T
        else:
            aux_y = array(y_test, ndmin=2)

        accuracy = sum(Y == aux_y) / (1.0 * len(y_test))

        return accuracy

    def gradient_descent(self, X, Y, U, error):
        if array(Y, ndmin=2).shape[1] != 1:
            aux_y = array(Y, ndmin=2).T
        else:
            aux_y = array(Y, ndmin=2)

        derivate = sum((aux_y * X) / (1 + exp(U)))
        return array(self.eta * derivate, ndmin=2).T

    def predict(self, u):
        Y_output = zeros((u.shape[0], 1))
        indices_j = where((1 - u) > u)
        indices_i = where(u > (1 - u))
        Y_output[indices_j] = -1
        Y_output[indices_i] = 1

        return Y_output
