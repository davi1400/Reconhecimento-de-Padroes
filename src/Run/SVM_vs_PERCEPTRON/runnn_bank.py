from sklearn.model_selection import KFold
from Strings import string
from sklearn.model_selection import train_test_split
from numpy import zeros, identity, array, concatenate, \
    ones, ravel, dot, where
from src.Utils.utils import get_accuracy, get_data, normalize
import cvxopt
import cvxopt.solvers
from src.Algorithms.Supervised.SupportVectorMachines import svm

if __name__ == '__main__':
    data = get_data("data_banknote_authentication.txt", type="csv")
    number_lines = data.shape[0]
    number_columns = data.shape[1]
    X = array(data, ndmin=2)[:, :number_columns - 1]
    Y = array(array(data, ndmin=2)[:, number_columns - 1], ndmin=2).T
    train_size = .8
    test_size = .2

    indices = where(Y == 0)
    Y[indices] = -1

    # print(X)
    # print(Y)

    for realization in range(10):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)

        data = array(data, ndmin=2)
        indices = where(data == 0)
        data[indices] = -1
        # data[:, :4] = normalize(data[:, :4])


        print(data)

        model = svm(data, type="HardSoft")
        best_w = model.train()

        accuracy_svm = model.test(x_test, y_test)
        print(accuracy_svm)