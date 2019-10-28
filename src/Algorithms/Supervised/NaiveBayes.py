from numpy import zeros, concatenate, array

class NaiveBayes:
    def __init__(self, Matrix, reviews, bag):
        """

        :param Matrix: Matrix de contagem com label no final
        """

        self.Matrix = Matrix
        self.reviews = reviews
        self.bag = bag

    def train(self):
        """

        :return:
        """

        '''
            First calculte the probability of each class
        '''
        N_words = len(self.bag)
        N_examples = self.Matrix.shape[0]
        vecor_y = self.Matrix[:, len(self.Matrix.T)-1:]
        P_y = zeros((int(max(vecor_y)[0]), 1))
        for class_y in vecor_y:
            P_y[int(class_y[0])-1] += 1

        P_y = P_y/N_examples

        """
            The probabilitys os each class is in P_y
            
            OBS:
                P(Y=1) is in position 0 of the vector
                P(Y=2) is in position 1 of the vector
                .
                .
                .
                P(Y=N) is in position N-1 of the vector
                
           P_xy = {
            word : {
                '1': p(word/'1'),
                '2': p(word/'2'),
                .
                .
                ,
                'N': p(word/N)
            }
            .
            .
            .
        }
        """
        P_xy = {}

        for i in range(len(self.bag)):
            word = self.bag[i]
            frequence_word_sentiment = concatenate((array(self.Matrix[:, i], ndmin=2).T,
                                                    self.Matrix[:, self.Matrix.shape[1]-1:]), axis=1)





        return P_y



