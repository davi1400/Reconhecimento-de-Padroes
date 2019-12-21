import cvxopt
from numpy import zeros, diag, array


class svm:
    def __init__(self, data, type=None):
        if type == "HardSoft":
            self.solver = self.quadratic_solver
        self.number_lines, self.number_columns = data.shape
        self.X = data[:, :self.number_columns]
        self.Y = data[:, self.number_columns]
        self.wheigths = zeros((self.number_columns, 1))

    def quadratic_solver(self):
        """
            The quadratic solver used is cvxopt, and his default form is:

            min (1/2)(x^T)PX + Q^TX
            st. Gx <= H
            Ax = b

            the form of svm hard soft is:

            min (1/2)W^TW
            st. Y(W^TX + b) >= 1 the same of -1. * Y(W^TX+ b) <= -1.* 1

            Q = zero matrix
            P = diagonal matrix of ones
            H = [[-1.]]
            W = X
            G = -1*Y*[[...X... 1 ]
                      [........1]
                            .
                            .
                            .
                      [.........1]]
        """

        Q = zeros((self.number_columns, 1))
        P = diag((self.number_columns, self.number_columns))
        H = array([[1]])
        G =

    def train(self):
        self.wheigths = self.solver()
