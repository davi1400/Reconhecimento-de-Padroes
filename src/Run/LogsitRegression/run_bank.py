from sklearn.model_selection import train_test_split
from numpy import zeros, concatenate, ones, where, array, argmin, log, exp, mean
from numpy.random import rand
from src.Utils.utils import sigmoid, get_data, normalize, get_confusion_matrix
from src.Algorithms.Supervised.LogisticRegression import LogisticRegression
from pandas import DataFrame

eta = 1e-1
ephocs = 1000
ERROR = []

if __name__ == '__main__':
    data = get_data("column_2C_weka.arff", type="arff")
    data = DataFrame(data)
    number_lines = data.shape[0]
    number_columns = data.shape[1]
    X = array(data, ndmin=2)[:, :number_columns - 1]
    Y = array(data, ndmin=2)[:, number_columns - 1]
    train_size = .8
    test_size = .2

    X = array(X, dtype=float)

    indices = where(Y == b'Abnormal')
    Y[indices] = -1

    indices = where(Y == b'Normal')
    Y[indices] = 1

    # indices = where(Y == 0)
    # Y[indices] = -1
    #
    # print(X)
    # print(Y)
    #
    # X = normalize(X)
    #
    # print(X)
    # print(Y)

    acc = []
    for realization in range(20):
        logistic = LogisticRegression(eta=eta, ephocs=ephocs)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        wheiths = logistic.train(x_train, y_train)
        accuracy = logistic.test(wheiths, x_test, y_test)
        acc.append(accuracy)

        y_output = logistic.test(wheiths, x_test, y_test, flag=True)
        confusion = get_confusion_matrix(y_output, y_test)
        print(confusion)
    print(mean(acc))
