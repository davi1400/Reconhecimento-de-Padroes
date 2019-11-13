import numpy as np
import matplotlib.pyplot as plt


class ColorMap:
    def __init__(self, X, Y, H=.02, mapa_cor=None, camadas_ocultas=1):
        self.X = X  # shape deve ser (n, 2)
        self.Y = Y
        self.H = H  # .02
        self.mapa_cor = mapa_cor
        self.camadas_ocultas = camadas_ocultas
        self.xx = None
        self.yy = None

    def map(self):
        x_min, x_max = self.X[:, 1].min() - 1, self.X[:, 1].max() + 1
        y_min, y_max = self.X[:, 2].min() - 1, self.X[:, 2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, self.H),
                             np.arange(y_min, y_max, self.H))
        new = np.c_[xx.ravel(), yy.ravel()]
        self.xx = xx
        self.yy = yy
        return new

    def coloring(self, G, weights, Flag=False, name="colorMap"):
        data = self.map()
        data = np.c_[-1 * np.ones(data.shape[0]), data]
        for camada in range(self.camadas_ocultas+1):
            Y_output = data.dot(weights)
            Y_predict = G(Y_output)
            if Flag:
                pos = self.X[np.where(self.Y == 1)[0]]
                neg = self.X[np.where(self.Y == -1)[0]]
            else:
                pos = self.X[np.where(self.Y == 1)[0]]
                neg = self.X[np.where(self.Y == 0)[0]]

            Z = Y_predict.reshape(self.xx.shape)

        plt.pcolormesh(self.xx, self.yy, Z, cmap=self.mapa_cor)
        plt.plot(pos[:, 1], pos[:, 2], 'bo', marker='s', markeredgecolor='w')
        plt.plot(neg[:, 1], neg[:, 2], 'ro', marker='s', markeredgecolor='w')
        plt.xlabel("X1")
        plt.ylabel("X2")

        plt.savefig(name)
        plt.show()











       #
       #  H = (Rede.Sigmoid(Rede.Pesos_ocultos.dot(np.c_[-1 * np.ones(new.shape[0]), new].T)).T)
       #  Z = (Rede.Sigmoid(Rede.Pesos_saida.dot((np.c_[-1 * np.ones(H.shape[0]), H]).T)))
       #
       #  Z = Rede.predicao(Z.T)
       #  Z = Z.
       #
        # plt.savefig('Grafico7Art8')
