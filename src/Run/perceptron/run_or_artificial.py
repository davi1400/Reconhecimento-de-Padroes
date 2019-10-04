"""
    Run the artificial dataset OR with perceptron algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
from numpy import concatenate, mean
from src.Utils.mock import generate_OR
from src.Utils.utils import normalize, heaveside
from src.Algorithms.Supervised.Perceptron import perceptron
from src.Utils.ColorMap import ColorMap
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    X, Y = generate_OR()
    p = perceptron(learning_rate=0.015, epochs=500, type=2, logist=True)
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

    print("OR accuracy:" + str(mean(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))





    # c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    # c.coloring(heaveside, weights)



    c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    c.coloring(heaveside, weights, name="ortestL")

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
    plt.savefig("AcuracysOR")
    plt.show()

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

    print("OR accuracy:" + str(mean(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))

    # c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    # c.coloring(heaveside, weights)



    c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    c.coloring(heaveside, weights, Flag=True, name="ortestT")

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
    plt.savefig("AcuracysORTH")
    plt.show()
