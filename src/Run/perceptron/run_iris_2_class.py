"""
    Before training IRIS we have to modife the expeted classes from 0,1,2 to 0,1. Because
    the perceptron is a binary classifier.
"""
import numpy as np
import pylab as pl
from src.Utils.utils import get_data
from src.Algorithms.Supervised.Perceptron import perceptron
import matplotlib.pyplot as plt
from numpy import reshape, array, mean
from sklearn.datasets import load_iris
from src.Utils.ColorMap import ColorMap
from matplotlib.colors import ListedColormap
from src.Utils.utils import heaveside,  normalize

#  just for the plot
IRIS = load_iris()

if __name__ == '__main__':
    data = get_data("Iris", type='csv')
    p = perceptron(data, 0.015, 100)
    p.X = normalize(p.X)
    p.add_bias()
    feature_names = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    target_names = array(['setosa', 'versicolor', 'virginica'], dtype='<U10')
    sepal_lenth = data[0]  # sepal length (cm)
    sepal_width = data[1]  # sepal width (cm)

    # plt.figure(figsize=(10, 7))
    #
    # plt.hist([sepal_lenth, sepal_width], bins=20, color=["green", "red"])
    # plt.title("sepal length (cm) vs sepal width (cm)")
    # plt.xlabel("sepal length (cm) vs sepal width (cm)")
    # plt.ylabel("Count")
    # plt.show()

    plots = [(0, 1), (0, 2), (0, 3)]

    formatter = plt.FuncFormatter(lambda i, *args: target_names[int(i)])

    plt.figure(figsize=(5, 4))
    plt.scatter(p.X[:, 1], p.X[:, 2], c=IRIS.target)
    plt.colorbar(ticks=[0, 1, 2], format=formatter)
    plt.xlabel(feature_names[0])  # sepal length (cm)
    plt.ylabel(feature_names[1])  # sepal width (cm)

    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # SETOSA VS OUTRAS
    accuracys = []
    confusions_matrix = []
    p.transform_binary("Iris-setosa")
    for realization in range(20):
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
    p.X = p.X[:, :3]
    p.transform_binary("Iris-setosa")
    print("Training two features - SETOSA width and SETOSA lenth")
    for realization in range(20):
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)

    print("SETOSA VS OUTRAS accuracy:" + str(mean(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))

    c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    c.coloring(heaveside, weights)


    # ------------------------------------------------------------------------------------------------------------------
    # VIRGINICA VS OUTRAS
    accuracys = []
    confusions_matrix = []
    p.transform_binary("Iris-virginica")
    for realization in range(20):
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
    p.transform_binary("Iris-versicolor")
    for realization in range(20):
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)

    print("VERSICOLOR VS OUTRAS accuracy:" + str(mean(accuracys)))
    print("Standard deviation of accuracy " + str(np.std(accuracys)))
    print("variance of accuracy " + str(np.var(accuracys)))

#  print(confusions_matrix)
