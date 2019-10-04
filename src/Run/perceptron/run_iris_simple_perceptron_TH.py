import numpy as np
import pylab as pl
from src.Utils.utils import get_data
from src.Algorithms.Supervised.Perceptron import perceptron
import matplotlib.pyplot as plt
from numpy import reshape, array, mean
from sklearn.datasets import load_iris
from src.Utils.ColorMap import ColorMap
from matplotlib.colors import ListedColormap
from src.Utils.utils import heaveside, normalize

if __name__ == '__main__':
    data = get_data("Iris", type='csv')
    p = perceptron(data, 0.015, 500, type=2)
    p.X = normalize(p.X)
    p.add_bias()
    """
        Utilizando sigmoid tangente hiperbolica
    """
    print("# ------------------------------------ #")
    print("Sigmoid TH")
    p.variances = []
    p.stds = []
    # SETOSA VS OUTRAS
    accuracys = []
    confusions_matrix = []
    p.transform_binary("Iris-setosa", flag=True)
    for realization in range(20):
        p.split()
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)

    print("SETOSA VS OUTRAS accuracy:" + str(mean(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))

    pl.matshow(confusion_matrix)
    pl.title('Matriz de confusao\n')
    pl.colorbar()
    plt.savefig('GraficoArt13')

    pl.show()

    plt.plot(range(20), accuracys)
    plt.xlabel("realizações")
    plt.ylabel("Acurácias")
    plt.savefig("AcuracysIris")
    plt.show()

    plt.plot(range(20), p.variances)
    plt.xlabel("ralizações")
    plt.ylabel("variancia")
    plt.savefig("variance iris setosaXoutras")
    plt.show()

    plt.plot(range(20), p.stds)
    plt.xlabel("ralizações")
    plt.ylabel("desvio padrão")
    plt.savefig("standard deviation setosaXoutras")
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # Training just two features
    accuracys = []
    confusions_matrix = []
    p.X = p.X[:, 3:]
    p.add_bias()
    p.transform_binary("Iris-setosa", flag=True)
    print("Training two features - SETOSA width and SETOSA lenth")
    for realization in range(20):
        p.split()
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)

    print("SETOSA VS OUTRAS accuracy:" + str(mean(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))

    c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    c.coloring(heaveside, weights, Flag=True)

    # ------------------------------------------------------------------------------------------------------------------
    # VIRGINICA VS OUTRAS
    accuracys = []
    confusions_matrix = []
    p.transform_binary("Iris-virginica", flag=True)
    for realization in range(20):
        p.split()
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)

    print("VIRGINICA VS OUTRAS accuracy:" + str(mean(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))

    # ------------------------------------------------------------------------------------------------------------------
    # VERSICOLOR VS OUTRAS
    accuracys = []
    confusions_matrix = []
    p.transform_binary("Iris-versicolor", flag=True)
    for realization in range(20):
        p.split()
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)

    print("VERSICOLOR VS OUTRAS accuracy:" + str(mean(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))

    #  print(confusions_matrix)


    # ---------------------------------------------------------------------------------------------------------------------
