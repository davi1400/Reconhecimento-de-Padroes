from sklearn.model_selection import train_test_split
from numpy import zeros, concatenate, ones, where, array, argmin, log, exp, mean
from numpy.random import rand
from src.Utils.utils import sigmoid, get_data, normalize
from src.Algorithms.Supervised.LogisticRegression import LogisticRegression

eta = 1e-1
ephocs = 1000
ERROR = []

if __name__ == '__main__':
    data = get_data("data_banknote_authentication.txt", type="csv")
    number_lines = data.shape[0]
    number_columns = data.shape[1]
    X = array(data, ndmin=2)[:, :number_columns - 1]
    Y = array(data, ndmin=2)[:, number_columns - 1]
    train_size = .8
    test_size = .2

    indices = where(Y == 0)
    Y[indices] = -1

    print(X)
    print(Y)

    X = normalize(X)

    print(X)
    print(Y)

    acc = []
    for realization in range(1):
        logistic = LogisticRegression(eta=eta, ephocs=ephocs)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        wheiths = logistic.train(x_train, y_train)
        accuracy = logistic.test(wheiths, x_test, y_test)
        acc.append(accuracy)
    print(mean(acc))
