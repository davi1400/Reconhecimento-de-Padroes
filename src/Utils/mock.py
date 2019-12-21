import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt


def genarete_AND():
    t = np.linspace(0, 2 * np.pi, 100)
    r = np.random.rand((100)) / 5.0
    x1, x2 = 1 + r * np.cos(t), 1 + r * np.sin(t)
    x3, x4 = r * np.cos(t), r * np.sin(t)
    x5, x6 = 1 + r * np.cos(t), r * np.sin(t)
    x7, x8 = r * np.cos(t), 1 + r * np.sin(t)
    data1 = np.concatenate([x1.T, x3.T, x5.T, x7.T], axis=0)
    data2 = np.concatenate([x2.T, x4.T, x6.T, x8.T], axis=0)
    data1 = np.array((data1), ndmin=2).T
    data2 = np.array((data2), ndmin=2).T
    X = np.concatenate([data1, data2], axis=1)
    Y = np.ones((X.shape[0], 1))
    Y[100:, ] = 0 * Y[100:, ]

    pos = X[np.where(Y == 1)[0]]
    neg = X[np.where(Y == 0)[0]]
    plt.plot(pos[:, 0], pos[:, 1], 'go')
    plt.plot(neg[:, 0], neg[:, 1], 'ro')

    # plt.savefig("../Relatórios/images/AND")

    plt.show()

    return [X, Y]


def generate_OR():
    t = np.linspace(0, 2 * np.pi, 100)
    r = np.random.rand((100)) / 5.0
    x1, x2 = 1 + r * np.cos(t), 1 + r * np.sin(t)
    x3, x4 = r * np.cos(t), r * np.sin(t)
    x5, x6 = 1 + r * np.cos(t), r * np.sin(t)
    x7, x8 = r * np.cos(t), 1 + r * np.sin(t)
    data1 = np.concatenate([x1.T, x3.T, x5.T, x7.T], axis=0)
    data2 = np.concatenate([x2.T, x4.T, x6.T, x8.T], axis=0)
    data1 = np.array((data1), ndmin=2).T
    data2 = np.array((data2), ndmin=2).T
    X = np.concatenate([data1, data2], axis=1)
    Y = np.ones((X.shape[0], 1))
    Y[100:200, ] = 0 * Y[100:200, ]

    pos = X[np.where(Y == 1)[0]]
    neg = X[np.where(Y == 0)[0]]
    plt.plot(pos[:, 0], pos[:, 1], 'go')
    plt.plot(neg[:, 0], neg[:, 1], 'ro')

    plt.show()

    return [X, Y]


def create_mock(N):
    x1 = [uniform(-1, 1), uniform(-1, 1)]
    x2 = [uniform(-1, 1), uniform(-1, 1)]

    angular_coefficient = ((x2[1] - x1[1]) * 1.) / ((x2[0] - x1[0]) * 1.)
    linear_coefficient = -1. * angular_coefficient * x1[0] + x1[1]

    if angular_coefficient * x2[0] + linear_coefficient == x2[1]:
        print("OK")
    else:
        return None, None

    data = np.zeros((N, 3))

    for i in range(N):
        X = [uniform(-1, 1), uniform(-1, 1)]
        y = angular_coefficient * X[0] + linear_coefficient - X[1]

        data[i, 0] = X[0]
        data[i, 1] = X[1]

        if y > 0:
            data[i, 2] = 1
        elif y < 0:
            data[i, 2] = -1

    out_of_data_points = np.zeros((N*10, 3))
    for i in range(N*10):
        X = [uniform(-1, 1), uniform(-1, 1)]
        y = angular_coefficient * X[0] + linear_coefficient - X[1]

        out_of_data_points[i, 0] = X[0]
        out_of_data_points[i, 1] = X[1]

        if y > 0:
            out_of_data_points[i, 2] = 1
        elif y < 0:
            out_of_data_points[i, 2] = -1

    return data, out_of_data_points


if __name__ == '__main__':
    verify = True

    while verify:
        data = create_mock(100)
        if data is None:
            data = create_mock(100)
        else:
            verify = False
    X = data[:, :2]
    Y = data[:, 2]
    pos = X[np.where(Y == 1)]
    neg = X[np.where(Y == -1)]

    plt.plot(pos[:, 0], pos[:, 1], 'r+')
    plt.plot(neg[:, 0], neg[:, 1], 'bo')
    plt.show()

    print(data)
