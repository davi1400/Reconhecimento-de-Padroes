"""
    Run the artificial dataset AND with perceptron algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from pandas import DataFrame
from numpy import concatenate, mean
from src.Utils.mock import genarete_AND
from src.Utils.utils import normalize, heaveside
from src.Algorithms.Supervised.Perceptron import perceptron
from src.Utils.ColorMap import ColorMap
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    X, Y = genarete_AND()
    p = perceptron(learning_rate=0.015, epochs=400, type=2, logist=True)
    p.X, p.Y = normalize(X), Y
    p.add_bias()

    accuracys = []
    confusions_matrix = []
    for realization in range(20):
        p.split()
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)

    print("AND accuracy:" + str(mean(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))


    # X_train, X_test, Y_train, Y_test = train_test_split(p.X, p.Y, test_size=0.2)
    # c = ColorMap(X_train, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    # c.coloring(heaveside, weights, name="andtraining")


    c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    c.coloring(heaveside, weights, name="andtestL")

    stdr = np.std(accuracys)


    # pl.matshow(confusion_matrix)
    # pl.title('Matriz de confusao\n')
    # pl.colorbar()
    # # plt.savefig('GraficoArt9')

    pl.show()

    plt.plot(range(20), accuracys)
    plt.xlabel("realizações")
    plt.ylabel("Acurácias")
    plt.savefig("AcuracysANDL")
    plt.show()

    # plt.plot(range(20), p.variances)
    # plt.xlabel("ralizações")
    # plt.ylabel("variancia")
    # plt.savefig("varianceand")
    # plt.show()
    #
    # plt.plot(range(20), p.stds)
    # plt.xlabel("ralizações")
    # plt.ylabel("desvio padrão")
    # plt.savefig("standard deviation and")
    # plt.show()


    # ----------------------------------------------------------------------------------------------------------------- #
    p.logistic = False
    p.variances = []
    p.stds = []
    for i in range(len(p.Y)):
        if p.Y[i] == 0:
            p.Y[i] = -1

    accuracys = []
    confusions_matrix = []
    for realization in range(20):
        p.split()
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)

    print("AND accuracy:" + str(mean(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))

    # c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    # c.coloring(heaveside, weights)



    c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    c.coloring(heaveside, weights, Flag=True, name="andtestT")

    # plt.plot(range(20), p.variances)
    # plt.xlabel("ralizações")
    # plt.ylabel("variancia")
    # plt.savefig("varianceor")
    # plt.show()
    #
    # plt.plot(range(20), p.stds)
    # plt.xlabel("ralizações")
    # plt.ylabel("desvio padrão")
    # plt.savefig("standard deviation or")
    # plt.show()

    plt.plot(range(20), accuracys)
    plt.xlabel("realizações")
    plt.ylabel("Acurácias")
    plt.savefig("AcuracysANDTH")
    plt.show()



