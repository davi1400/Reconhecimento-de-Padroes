import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.Utils.utils import normalize
from src.Utils.ColorMap import ColorMap
from src.Algorithms.Supervised.LogisticRegression import LogisticRegression
from matplotlib.colors import ListedColormap
from scipy.special import expit
pi = math.pi


def points_in_circles(r, z=2, raio=20):
    out = []
    for i in range(-r * z, r * z, 1):
        for j in range(-r * z, r * z, 1):
            if (math.sqrt((i / z) ** 2 + (j / z) ** 2) <= raio):
                out.append(((i / z), (j / z)))

    return out


if __name__ == '__main__':

    points = points_in_circles(100, z=1)
    points2 = points_in_circles(100, z=1, raio=10)

    for i in points2:
        points.pop(points.index(i))
    x = []
    y = []
    for point in points:
        x.append(point[0])
        y.append(point[1])

    for i in x:
        x[x.index(i)] = x[x.index(i)] + np.random.rand()
    for i in y:
        y[y.index(i)] = y[y.index(i)] + np.random.rand()

    for i in range(int(len(x) / 4)):
        x.pop(np.random.randint(0, len(x)))

    for i in range(int(len(y) / 4)):
        y.pop(np.random.randint(0, len(y)))
    x1 = np.add(x[:int(len(x) // 2)], -2)
    y1 = np.subtract(y[:int(len(x) // 2)], -5.5)
    z1 = [1] * len(x1)

    matrix1 = np.array(list(zip(y1, x1, z1)))

    x2 = np.add(x[int(len(x) // 2):], 2)
    y2 = np.subtract(y[int(len(x) // 2):], 5.5)
    z2 = [-1] * len(x2)

    matrix2 = np.array(list(zip(y2, x2, z2)))

    plt.scatter(y1, x1, c='darkturquoise')
    plt.scatter(y2, x2, c='sandybrown')
    plt.savefig('GraficoArtY')
    plt.show()

    data = np.concatenate((matrix1, matrix2), axis=0)
    np.random.shuffle(data)
    print(data)
    print(data.shape)

    logistic = LogisticRegression(eta=1e-3, ephocs=1000)
    accuracys = []

    for realization in range(10):
        X_train, X_test, Y_train, Y_test = train_test_split(data[:, :2], data[:, 2:], test_size=0.2)
        W = logistic.train(X_train, Y_train)
        accuracys.append(logistic.test(W, X_test, Y_test))

    print(np.mean(accuracys))

    point = np.array([10, -10], ndmin=2)
    y = np.array([1], ndmin=2)
    print(logistic.test(W, point, y))

    plot_x = np.array([np.min(X_test[:, 0] - 10), np.max(X_test[:, 1]) + 10])
    plot_y = - 1 / W[2] * (W[1] * plot_x + W[0])
    plt.scatter(y1, x1, c='darkturquoise')
    plt.scatter(y2, x2, c='sandybrown')
    plt.plot(plot_x, plot_y, color='k', linewidth=2)
    plt.savefig('GraficoArtX')
    plt.show()

    Mapa_Cor = ListedColormap(['#F4A460', '#00ced1'])
    c = ColorMap(X_test, Y_test, mapa_cor=Mapa_Cor)
    c.coloring(expit, W, Flag=True, colors=['co', '#F4A460'])