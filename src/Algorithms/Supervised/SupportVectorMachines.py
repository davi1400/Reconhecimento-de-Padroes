from numpy import zeros, identity, array, concatenate, ones, multiply, matrix, outer, ravel, dot, arange
from src.Utils.utils import get_accuracy
from math import sqrt
import cvxopt
import cvxopt.solvers


class svm:
    def __init__(self, data, type=None):
        if type == "HardSoft":
            self.solver = self.quadratic_solver
        self.number_lines, self.number_columns = data.shape
        self.X = data[:, :self.number_columns-1]
        self.Y = data[:, self.number_columns-1]
        self.best_wheigths = zeros((self.number_columns, 1))
        self.a = 0
        self.sv = 0
        self.vec_sup = 0
        self.sv_y = 0

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
                            .cvxopt.msk
                            .
                            .
                      [.........1]]HardSoft
        """

        P = self.get_P()
        Q = self.get_Q()
        G = self.get_G()
        H = self.get_H()

        try:
            solution = cvxopt.solvers.qp(P, Q, G, H)
        except ValueError as error:
            return None

        self.best_wheigths = ravel(solution['x'])

        return self.best_wheigths

    def get_P(self):
        tmp1 = concatenate((ones((self.number_columns - 1, 1)), zeros((self.number_columns - 1, 1))), axis=1)
        tmp2 = zeros((1, self.number_columns - 1))
        tmp3 = concatenate((tmp1, tmp2))
        tmp4 = zeros((self.number_columns, 1))
        tmp5 = concatenate((tmp3, tmp4), axis=1)
        P = cvxopt.matrix(tmp5)

        return P

    def get_Q(self):
        tmp1 = zeros((self.number_columns, 1))
        Q = cvxopt.matrix(tmp1)

        return Q

    def get_G(self):
        tmp2 = array(self.Y, ndmin=2).T * concatenate((self.X, ones((self.number_lines, 1))), axis=1)
        G = cvxopt.matrix(tmp2 * -1.)

        return G

    def get_H(self):
        tmp2 = ones((self.number_lines, 1))
        H = cvxopt.matrix(tmp2 * -1.)

        return H

    def train(self):
        solution = self.solver()
        return solution

    def predict(self, H, domain=None):
        if domain == [-1., 1.]:
            for i in range(len(H)):
                if H[i][0] > 0:
                    H[i][0] = 1
                else:
                    H[i][0] = -1

            return H

    def get_foward(self, X_test):
        H_output = dot(X_test, array(self.best_wheigths[:2], ndmin=2).T) + self.best_wheigths[2]
        self.vec_sup = sum(H_output == 1.)
        return H_output

    def test(self, X_test, y_test):
        H_output = self.get_foward(X_test)
        Y_output = self.predict(H_output, domain=[-1., 1.])

        accuracy = get_accuracy(Y_output, array(y_test, ndmin=2).T)

        return accuracy

    def get_number_support_vectors(self):
        # Support vectors have non zero lagrange multipliers
        # sv = self.best_wheigths > 1e-5
        # ind = arange(len(self.best_wheigths))[sv]
        # self.a = self.best_wheigths[sv]
        # self.sv = self.X[ind]
        # self.sv_y = self.Y[ind]
        # print("%d support vectors out of %d points" % (len(self.a), self.number_lines))

        index = 0
        for point in self.X:
            x = concatenate((point, [1]), axis=0)
            distance = dot(array(self.best_wheigths, ndmin=2), array(x, ndmin=2).T)
            norm_w = dot(array(self.best_wheigths, ndmin=2), array(self.best_wheigths, ndmin=2).T)
            distance = abs(distance) / sqrt(norm_w)

            if distance == 1.:
                self.vec_sup += 1

        print(self.vec_sup)



