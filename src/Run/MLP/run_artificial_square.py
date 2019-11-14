import math
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from src.Utils.utils import normalize
from src.Algorithms.Supervised.MLP import MultiLayerPerceptron
from matplotlib.colors import ListedColormap
pi = math.pi


def points_in_square(r, z=2, origin=(0, 0)):
    out = []
    for i in range(-r * z, r * z, 1):
        for j in range(-r * z, r * z, 1):
            out.append(((i / z) + origin[0], (j / z) + origin[1]))
    return out


def list_of_tuple_to_np(lp, ndim=2):
    out = []
    for i in lp:
        point = []
        for j in range(ndim):
            point.append(i[j])
        out.append(point)
    return np.array(out)


if __name__ == '__main__':


    points = points_in_square(20, z=1, origin=[13, -13])
    points2 = points_in_square(20, z=1, origin=[-13, 13])

    points = list_of_tuple_to_np(points)
    points2 = list_of_tuple_to_np(points2)
    tam = len(points)
    points = list(points)
    points2 = list(points2)

    for i in range(tam // 3):
        random_index = np.random.randint(0, len(points))
        points.pop(random_index)
    for i in range(tam // 3):
        random_index = np.random.randint(0, len(points2))
        points2.pop(random_index)

    for i in range(len(points2)):
        points2[i] = points2[i] + [np.random.randint(-3, 3), np.random.randint(-3, 3)]
    for i in range(len(points)):
        points[i] = points[i] + [np.random.randint(-3, 3), np.random.randint(-3, 3)]



    points = np.array(points)
    points2 = np.array(points2)

    points_new = np.concatenate((points, -1. * np.ones((points.shape[0], 1))), axis=1)
    points_new2 = np.concatenate((points2, np.ones((points2.shape[0], 1))), axis=1)
    data = np.concatenate((points_new, points_new2), axis=0)
    np.random.shuffle(data)

    plt.scatter(points[:, 0], points[:, 1], c='darkturquoise')
    plt.scatter(points2[:, 0], points2[:, 1], c='sandybrown')

    plt.show()
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
        Rede.Train(X_train, Y_train, 300)

        G_SaidaTest = Rede.Saida(X_test)
        Y_SaidaTest = Rede.predicao(Y_test)
        print(((G_SaidaTest == Y_SaidaTest).sum()) / (1.0 * len(Y_SaidaTest)))

    # ------------------------------------------------------------------------------------------------
    # Plot
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