from numpy import zeros, concatenate, array, where, log, argmax, inf


class NaiveBayes:
    def __init__(self, Matrix, reviews, bag, type=None):
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

    def train(self):
        if self.type == "bernolli":
            return self.train_bernolli()
        if self.type == "gaussian":
            return self.train_gaussian()

    def train_gaussian(self, train):

        N_class = max(train[:, train.shape[0] - 1:])
        Matrix_Mean = zeros((train.shape[1], N_class)).T
        Variance_Matrix = zeros((train.shape[1], N_class)).T

        for c in N_class:
            N_y = len(where(train[:, train.shape[0]-1:] == c)[0])
            for i in range(train.shape[1]):
                indices = where(train[:, train.shape[0] - 1:] == c)
                Matrix_Mean[c][i] = (1./(1.*N_y))*sum(train[indices])

    def train_bernolli(self):
        """

        :return:
        """

        '''
            First calculte the probability of each class
        '''
        N_class = int(max(self.Matrix[:, self.Matrix.shape[1]-1:])[0])
        N_words = len(self.bag)
        N_examples = self.Matrix.shape[0]
        vecor_y = self.Matrix[:, len(self.Matrix.T)-1:]
        thetas_c = zeros((int(max(vecor_y)[0]), 1))
        for class_y in vecor_y:
            thetas_c[int(class_y[0])-1] += 1

        thetas_c = thetas_c/N_examples

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
                                                    self.Matrix[:, self.Matrix.shape[1]-1:]), axis=1)

            for c in range(N_class):
                indices = where(frequence_word_sentiment[:, 1:] == c+1)
                theta_word_c = (sum(frequence_word_sentiment[indices]))/(1.0*len(indices[0]))
                if theta_word_c == 0.0:
                    theta_word_c = 1e-5
                thetas_ic[i][c] = theta_word_c

        return thetas_ic, thetas_c

    def test(self, row, thetas_ic, thetas_c):
        _len = len(row)
        P_yx = zeros((thetas_c.shape[0], 1))

        for c in range(thetas_c.shape[0]):
            P_y = log(thetas_c[c][0])
            for i in range(_len-1):
                theta_ic = thetas_ic[i, c]
                x_i = row[i]
                if self.bernolli(theta_ic, x_i) == -1.0*inf:
                    continue
                P_yx[c] += self.bernolli(theta_ic, x_i)
            P_yx += P_y

        return P_yx

    def predict(self, p):
        return argmax(p)

    def bernolli(self, theta_i, x_i):
        if x_i > 1.:
            x_i = 1.0
        return x_i*log(theta_i) + (1.0-x_i)*log(1.0-theta_i)

    def gausian(self):
        pass



