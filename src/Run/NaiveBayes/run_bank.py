
from numpy import zeros, concatenate, array, where, log, argmax, inf, sqrt, pi, exp, mean
from Strings import string
from sklearn.model_selection import train_test_split
from numpy.random import rand
from src.Utils.utils import sigmoid, get_data, get_confusion_matrix, plot_confusion_matrix
from sklearn.model_selection import KFold
from src.Algorithms.Supervised.NaiveBayes import NaiveBayes


if __name__ == '__main__':
    data = get_data("data_banknote_authentication.txt", type="csv")
    number_lines = data.shape[0]
    number_columns = data.shape[1]
    X = array(data, ndmin=2)[:, :number_columns - 1]
    Y = array(array(data, ndmin=2)[:, number_columns - 1], ndmin=2).T
    train_size = .8
    test_size = .2

    indices = where(Y == -1)
    Y[indices] = 0

    print(X)
    print(Y)

    acc = []
    acc_eta = []
    kf = KFold(n_splits=10)
    eta_validation_vector = [1., 0.15, 0.1, 1e-2, 1e-3]
    for realization in range(5):
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        naive = NaiveBayes(type="gaussian")

        # for eta_val in eta_validation_vector:
        #     acc_validation = []
        #     for train_index, test_index in kf.split(x_train):
        #         # print("TRAIN:", train_index, "TEST:", test_index)
        #         x_train_val, x_test_val = x_train[train_index], x_train[test_index]
        #         y_train_val, y_test_val = y_train[train_index], y_train[test_index]
        #
        #         wheiths_val = train(x_train_val, y_train_val)
        #         accuracy = test(wheiths_val, x_test_val, y_test_val)
        #
        #         acc_validation.append(accuracy)
        #     acc_eta.append(mean(acc_validation))
        #
        #
        # indice = argmax(acc_eta)

        matrix_mean, variance_matrix, thetas_c = naive.train(x_train, y_train)
        accuracy = naive.test(x_test=x_test, y_test=y_test, Matrix_Mean=matrix_mean, Variance_Matrix=variance_matrix,
                              thetas_c=thetas_c)
        acc.append(accuracy)

        # matrix_Mean, variance_Matrix, thetas_c = train(x_train, y_train)
        # accuracy = test(x_test, y_test, matrix_Mean, variance_Matrix, thetas_c)
        # y_output = test(x_test, y_test, matrix_Mean, variance_Matrix, thetas_c, flag=True)
        # acc.append(accuracy)

        # confusion_matrix = get_confusion_matrix(y_output, y_test)
        print(string.RUN.format(realization, 1000, accuracy))

    # plot_confusion_matrix(confusion_matrix)
    # print(mean(acc))