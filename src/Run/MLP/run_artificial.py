import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from src.Utils.utils import normalize
from src.Algorithms.Supervised.MLP import MultiLayerPerceptron
from matplotlib.colors import ListedColormap

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
    x1 = np.add(x[:int(len(x) // 2)], 2)
    y1 = np.subtract(y[:int(len(x) // 2)], -7.5)
    z1 = [1] * len(x1)

    matrix1 = np.array(list(zip(y1, x1, z1)))

    x2 = np.add(x[int(len(x) // 2):], -2)
    y2 = np.subtract(y[int(len(x) // 2):], 7.5)
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

    X = np.zeros((int(data.shape[0]), len(data[0]) - 1))

    for i in range(int(data.shape[0])):
        for j in range(len(data[0]) - 1):
            X[i][j] = data[i][j]

    Y = np.zeros((int(data.shape[0]), 1))

    Mat_Y = np.zeros((Y.shape[0], 2))
    for i in range(len(Y)):
        if data[:, 2][i] == -1:
            Mat_Y[i, 0] = 1
        elif data[:, 2][i] == 1:
            Mat_Y[i, 1] = 1

    for realizacoes in range(1):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Mat_Y, test_size=0.2)
        Rede = MultiLayerPerceptron(X_train.shape[1], 12, 2, 0.15, False)
        Rede.InicializacaoPesos()
        Rede.Train(X_train, Y_train, 500)

        G_SaidaTest = Rede.Saida(X_test)
        Y_SaidaTest = Rede.predicao(Y_test)
        print(((G_SaidaTest == Y_SaidaTest).sum()) / (1.0 * len(Y_SaidaTest)))

    h = .02
    Mapa_Cor = ListedColormap(['#FFAAAA', '#AAAAFF'])
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    new = np.c_[xx.ravel(), yy.ravel()]
    #

    Z = Rede.Saida(new)
    pos = X_test[np.where(Rede.predicao(Y_test) == 1)[0]]
    neg = X_test[np.where(Rede.predicao(Y_test) == 0)[0]]
    # Z = Rede.predicao(Z.T)
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=Mapa_Cor)
    plt.plot(pos[:, 0], pos[:, 1], 'bo', marker='s', markeredgecolor='w')
    plt.plot(neg[:, 0], neg[:, 1], 'ro', marker='s', markeredgecolor='w')
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()