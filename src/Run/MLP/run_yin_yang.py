import math
import numpy as np
import matplotlib.pyplot as plt

pi = math.pi


def list_of_tuple_to_np(lp, ndim=2):
    out = []
    for i in lp:
        point = []
        for j in range(ndim):
            point.append(i[j])
        out.append(point)
    return np.array(out)


def points_in_circles(z=2, raio=20, origin=(0, 0)):
    r = raio
    out = []
    for i in range(-r * z, r * z, 1):
        for j in range(-r * z, r * z, 1):
            sqrt = (i / z) * 2 + (j / z) * 2
            if sqrt < 0:
                print(sqrt)
            # if (i/z)*2 + (j/z)*2 > 0:
            if (math.sqrt((i / z) ** 2 + (j / z) ** 2) < raio):
                out.append(((i / z) + origin[0], (j / z) + origin[1]))
            # else:
            #     sqrt = -1.*((i/z)*2 + (j/z)*2)
            #     if (math.sqrt(sqrt) < raio):
            #         out.append(((i / z) + origin[0], (j / z) + origin[1]))

    return out


def points_in_square(r, z=2, origin=(0, 0)):
    out = []

    for i in range(-r * z, r * z, 1):
        for j in range(-r * z, r * z, 1):
            out.append(((i / z) + origin[0], (j / z) + origin[1]))
    return out


def plot_list_tuples(lp, c="darkturquoise", x_separator=None, y_separator=None, left_color=None, right_color=None, flag=False):
    lp = list_of_tuple_to_np(lp)
    if left_color or right_color:
        def f(x):
            return x[0] < x_separator

        def f2(x):
            return not x[0] < x_separator

        lpl = np.array(list(filter(f, lp)))
        if not flag:
            plt.scatter(lpl[:, 0], lpl[:, 1], c=left_color)
        lpr = np.array(list(filter(f2, lp)))
        if not flag:
            plt.scatter(lpr[:, 0], lpr[:, 1], c=right_color)
    else:
        if not flag:
            plt.scatter(lp[:, 0], lp[:, 1], c=c)
    if flag:
        return lpl, lpr


if __name__ == '__main__':
    circle1 = points_in_circles(z=3, raio=3, origin=(0, 10))
    circle2 = points_in_circles(z=3, raio=3, origin=(0, -10))
    circle21 = points_in_circles(z=3, raio=10, origin=(0, 10))
    circle3 = points_in_circles(z=3, raio=20, )
    circle31 = points_in_circles(z=3, raio=10, origin=(0, -10))

    lpr, lpl = plot_list_tuples(circle3, c='b', x_separator=0, left_color='b', right_color='r', flag=True)

    # plot_list_tuples(circle21, c='b')
    # plot_list_tuples(circle31, c='r')

    red_points = np.concatenate((lpl, np.array(circle21, ndmin=2)), axis=0)
    blue_points = np.concatenate((lpr, np.array(circle31, ndmin=2)), axis=0)
    # aux_blue = np.zeros((blue_points.shape[0] - (np.array(circle21, ndmin=2)).shape[0], 2))
    # blue_subtract = np.concatenate((np.array(circle21, ndmin=2), aux_blue), axis=0)

    for point in circle21:
        blue_points = (blue_points.tolist()).pop(point)

    # blue_points = np.subtract(blue_points, blue_subtract)
    # blue_add = np.zeros((red_points.shape[0] - blue_points.shape[0], 2))
    # blue_points = np.concatenate((blue_points, blue_add), axis=0)
    # blue_points = np.intersect1d(blue_points, red_points)

    # blue_points = list(blue_points.intersection(red_points))
    # red_points = list(red_points)

    # red_points =
    # red_points.union(set(circle21))
    # blue_points = set(lpr)
    # blue_points.intersection(set(circle21))
    # blue_points.union(set(circle31))
    #
    # red_points = list(red_points)
    # blue_points = list(blue_points)




    # plot_list_tuples(circle1, c='r')
    # plot_list_tuples(circle2, c='b')

    # plt.plot(blue_points[:, 0], blue_points[:, 1], 'bo')
    # plt.plot(red_points[:, 0], red_points[:, 1], 'r*')

    plot_list_tuples(red_points, c='r')
    plot_list_tuples(blue_points, c='b')

    plt.xlim(-40, 40)
    plt.ylim(-40, 40)
    plt.show()

    # red_points = np.array(circle1, ndmin=2)
    # red_points = np.concatenate((red_points, lpr, np.array(circle31, ndmin=2)), axis=0)
    # red_points = np.concatenate((red_points, np.ones((red_points.shape[0], 1))), axis=1)

    # blue_points = np.array(circle2, ndmin=2)
    # blue_points = np.concatenate((blue_points, lpl, np.array(circle21, ndmin=2)), axis=0)
    # blue_points = np.concatenate((blue_points, -1.0 * np.ones((blue_points.shape[0], 1))), axis=1)

    # plt.plot(red_points, 'ro')
    # plt.plot(blue_points, 'b*')

    # points = np.concatenate((red_points, blue_points), axis=0)
    #
    # red = points[np.where(points[:, 2] == 1)]
    # blue = points[np.where(points[:, 2] == -1)]
    #

    #
    # plt.xlim(-40, 40)
    # plt.ylim(-40, 40)
    # plt.show()
