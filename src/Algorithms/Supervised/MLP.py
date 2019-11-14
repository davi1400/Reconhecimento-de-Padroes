import numpy as np
from scipy.special import expit
import matplotlib.pyplot as plt


# TODO revisar essa classe antiga que eu fiz
class MultiLayerPerceptron:
    def __init__(self, N_Padroes, N_Neruronios, N_Classes, lr, Regressao=False):
        self.N_Padroes = int(N_Padroes)
        self.N_Neruronios = int(N_Neruronios)
        self.N_Classes = int(N_Classes)
        self.lr = lr
        self.key = Regressao
        self.Pesos_saida = np.random.rand(self.N_Classes, self.N_Neruronios + 1)  # (cxH)
        self.Pesos_ocultos = np.random.rand(self.N_Neruronios, self.N_Padroes)  # (Hxp)

    def predicao(self, Y):
        y = np.zeros((Y.shape[0], 1))
        for j in range(Y.shape[0]):
            i = np.where(Y[j, :] == Y[j, :].max())[0][0]
            y[j] = i
        return y

    def Sigmoid(self, h):
        return expit(h)

    def InicializacaoPesos(self):
        # Criação dos pesos
        pass

    def Saida(self, X):
        # 1. Fase de Propagação
        H_Oculto = self.Sigmoid(self.Pesos_ocultos.dot(X.T))  # (Hxn)
        if self.N_Classes > 1:
            G_Saida = (
                self.Sigmoid(self.Pesos_saida.dot((np.c_[-1 * np.ones(H_Oculto.shape[1]), H_Oculto.T]).T))).T  # (cxn)
            return self.predicao(G_Saida)
        # Apenas uma classe é regressão
        elif self.N_Classes == 1 and self.key == True:
            G_Saida = (self.Pesos_saida.dot((np.c_[-1 * np.ones(H_Oculto.shape[1]), H_Oculto.T]).T)).T  # (cxn)

        return G_Saida

    def Train(self, X_train, MatY_train, epocas):
        for ep in range(epocas):
            r = np.random.permutation(X_train.shape[0])
            for k in range(len(r)):
                exemploX = np.array((X_train[r[k]]), ndmin=2).T  # (px1)
                exemploY = np.array((MatY_train[r[k]]), ndmin=2).T  # (cx1)

                # 1. Fase de Propagação
                H_Oculto = self.Sigmoid(self.Pesos_ocultos.dot(exemploX))  # (Hx1)
                if self.N_Classes > 1 and self.key == False:
                    G_saida = self.Sigmoid(self.Pesos_saida.dot((np.c_[-1, H_Oculto.T]).T))  # (cx1)
                else:
                    G_saida = (self.Pesos_saida.dot((np.c_[-1, H_Oculto.T]).T))  # (cx1)

                # 2. Propagação do erro na camada oculta
                Error_Saida = exemploY - G_saida

                if not self.key:
                    Error_Oculto = self.Pesos_saida.T.dot((G_saida * (1 - G_saida)) * Error_Saida)  # (Hx1)
                    self.Pesos_saida += self.lr * (
                        ((G_saida * (1 - G_saida)) * Error_Saida).dot((np.c_[-1, H_Oculto.T])))
                else:
                    Error_Oculto = self.Pesos_saida.T.dot(1 * Error_Saida)  # (Hx1)
                    self.Pesos_saida += self.lr * (1 * Error_Saida.dot((np.c_[-1, H_Oculto.T])))

                self.Pesos_ocultos += self.lr * (((H_Oculto * (1 - H_Oculto)) * Error_Oculto[1:, :]).dot(exemploX.T))
