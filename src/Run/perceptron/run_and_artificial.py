"""
    Run the artificial dataset AND with perceptron algorithm
"""
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from numpy import concatenate, mean
from src.Utils.mock import genarete_AND
from src.Utils.utils import normalize, heaveside
from src.Algorithms.Supervised.Perceptron import perceptron
from src.Utils.ColorMap import ColorMap
from matplotlib.colors import ListedColormap


if __name__ == '__main__':
    X, Y = genarete_AND()
    p = perceptron(learning_rate=0.001, epochs=1000)
    p.X, p.Y = normalize(X), Y
    p.add_bias()

    accuracys = []
    confusions_matrix = []
    for realization in range(20):
        weights, X_test, Y_test = p.train()
        accuracy, confusion_matrix = p.test(weights, X_test, Y_test, confusion_matrix=True)
        accuracys.append(accuracy)
        confusions_matrix.append(confusion_matrix)

    print("AND accuracy:" + str(mean(accuracys)))


    c = ColorMap(X_test, Y_test, mapa_cor=ListedColormap(['#FFAAAA', '#AAAAFF']))
    c.coloring(heaveside, weights)



