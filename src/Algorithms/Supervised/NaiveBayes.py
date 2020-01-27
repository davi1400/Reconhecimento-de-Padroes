from numpy import zeros, concatenate, array, where, log, argmax, inf, sqrt, pi, exp, ndarray


class NaiveBayes:
    def __init__(self, Matrix=None, reviews=None, bag=None, type=None):
        """
        Obs: The indidice of the bag that contais the word i, is the same for the matriz, that contais the word j, but
        in the matriz j is a colummn

        Matrix is for training
        :param Matrix: Matrix de contagem com label no final
        """

        self.Matrix = Matrix
        self.reviews = reviews
        self.bag = bag
        self.type = type

    def train(self, x_train=None, y_train=None):
        if self.type == "bernolli":
            return self.train_bernolli()
        if self.type == "gaussian":
            return self.train_gaussian(x_train, y_train)

    def test(self, x_test=None, y_test=None, row=None, thetas_ic=None, thetas_c=None, Matrix_Mean=None, Variance_Matrix=None):
        if self.type == "bernolli":
            return self.test_bernolli(row, thetas_ic, thetas_c)
        if self.type == "gaussian":
            return self.test_gaussian(x_test, y_test, Matrix_Mean, Variance_Matrix, thetas_c)

    def train_gaussian(self, x_train, y_train):

        if isinstance(max(y_train), ndarray):
            max_class = max(y_train)[0]
        else:
            max_class = max(y_train)
        N_class = max_class + 1
        N_examples = x_train.shape[0]
        thetas_c = zeros((int(N_class), 1))

        for c in y_train:
            if isinstance(c, ndarray):
                indice = c[0]
            else:
                indice = c
            thetas_c[int(indice)] += 1

        thetas_c = thetas_c / N_examples

        Matrix_Mean = zeros((x_train.shape[1], int(N_class))).T
        Variance_Matrix = zeros((x_train.shape[1], int(N_class))).T

        for c in range(int(N_class)):
            N_y = len(where(y_train == c)[0])
            for i in range(x_train.shape[1]):
                indices = where(y_train == c)[0]
                Matrix_Mean[c][i] = (1. / (1. * N_y)) * sum(x_train[:, i][indices])
                Variance_Matrix[c][i] = (1. / (1. * N_y)) * sum((x_train[:, i][indices] - Matrix_Mean[c][i]) ** 2)

        return Matrix_Mean, Variance_Matrix, thetas_c

    def test_gaussian(self, x_test, y_test, Matrix_Mean, Variance_Matrix, thetas_c):

        cont = 0

        for j in range(x_test.shape[0]):
            x_example = x_test[j]
            y_example = y_test[j]

            P_yx = zeros((Matrix_Mean.shape[0], 1))

            for c in range(Matrix_Mean.shape[0]):
                p_y = log(thetas_c[c])
                for i in range(Matrix_Mean.shape[1]):
                    X_i = x_example[i]
                    P_yx[c] += log(self.gaussian(X_i, Variance_Matrix[c][i], Matrix_Mean[c][i]))
                P_yx[c] += p_y

            Y = self.predict(P_yx)
            if Y == y_example:
                cont += 1

        return cont / (1.0 * j)

    def train_bernolli(self):
        """

        :return:
        """

        '''
            First calculte the probability of each class
        '''
        N_class = int(max(self.Matrix[:, self.Matrix.shape[1] - 1:])[0])
        N_words = len(self.bag)
        N_examples = self.Matrix.shape[0]
        vecor_y = self.Matrix[:, len(self.Matrix.T) - 1:]
        thetas_c = zeros((int(max(vecor_y)[0]), 1))
        for class_y in vecor_y:
            thetas_c[int(class_y[0]) - 1] += 1

        thetas_c = thetas_c / N_examples

        """
            The probabilitys os each class is in P_y
            
            OBS:
                P(Y=1) is in position 0 of the vector
                P(Y=2) is in position 1 of the vector
                .
                .
                .
                P(Y=N) is in position N-1 of the vector
                
        """
        P_xy = {}
        thetas_ic = zeros((len(self.bag), N_class))
        for i in range(len(self.bag)):
            frequence_word_sentiment = concatenate((array(self.Matrix[:, i], ndmin=2).T,
                                                    self.Matrix[:, self.Matrix.shape[1] - 1:]), axis=1)

            for c in range(N_class):
                indices = where(frequence_word_sentiment[:, 1:] == c + 1)
                theta_word_c = (sum(frequence_word_sentiment[indices])) / (1.0 * len(indices[0]))
                if theta_word_c == 0.0:
                    theta_word_c = 1e-5
                thetas_ic[i][c] = theta_word_c

        return thetas_ic, thetas_c

    def test_bernolli(self, row, thetas_ic, thetas_c):
        _len = len(row)
        P_yx = zeros((thetas_c.shape[0], 1))

        for c in range(thetas_c.shape[0]):
            P_y = log(thetas_c[c][0])
            for i in range(_len - 1):
                theta_ic = thetas_ic[i, c]
                x_i = row[i]
                if self.bernolli(theta_ic, x_i) == -1.0 * inf:
                    continue
                P_yx[c] += self.bernolli(theta_ic, x_i)
            P_yx += P_y

        return P_yx

    def predict(self, p):
        return argmax(p)

    def bernolli(self, theta_i, x_i):
        if x_i > 1.:
            x_i = 1.0
        return x_i * log(theta_i) + (1.0 - x_i) * log(1.0 - theta_i)

    def gaussian(self, X_i, Variance_i_c, Mean_i_c):
        Fraction = (1. / sqrt(2 * pi * Variance_i_c))
        Exponential = (-1. / 2.) * ((X_i - Mean_i_c) / sqrt(Variance_i_c)) ** 2

        return Fraction * exp(Exponential)

