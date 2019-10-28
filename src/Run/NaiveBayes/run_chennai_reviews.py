from numpy import zeros, where, nan, array
from random import shuffle
from src.Utils.SentimentAnalisys import SentimentAnalisys
from sklearn.model_selection import train_test_split
from src.Utils.utils import get_data
from src.Algorithms.Supervised.NaiveBayes import NaiveBayes


def pre_processing(reviews, bag):
    Matrix = zeros((len(reviews), len(bag)+1))
    i = 0
    for review, sentiment in reviews:
        if str(sentiment) != 'nan' and len(sentiment.split()) == 1:
            Matrix[i] = SentimentAnalisys.text_to_vector(Matrix[i], review, int(sentiment), bag, i)
        i += 1

    return Matrix


if __name__ == '__main__':
    data = get_data('chennai_reviews.csv', type='csv')

    accuracys = []

    for realization in range(1):
        cont = 0
        reviews_train, reviews_test = train_test_split(array(data[[2, 3]])[1:], test_size=0.2)
        bag_train = list(SentimentAnalisys.create_vocabulary(reviews_train[:, 0]))
        Matrix_train = pre_processing(reviews_train, bag_train)
        naive = NaiveBayes(Matrix_train, reviews_train, bag_train)
        thetas_ic, thetas_c = naive.train()

        # ----------------------------------------------------------------------------------------------------------- #
        # test
        l = 0
        for review, sentiment in reviews_test:
            if str(sentiment) != 'nan' and len(sentiment.split()) == 1:
                text_filtered = list(SentimentAnalisys.filter_stopwords(review))

                thetas = [thetas_c[int(sentiment)-1][0]]
                x = []
                for word in text_filtered:
                    indice_word = where(array(bag_train, ndmin=1) == word)
                    if word in bag_train:
                        x.append(1)
                    else:
                        x.append(0)

                    if len(indice_word[0]) > 0:
                        thetas.append(list(thetas_ic[indice_word][0]))

                        p = naive.test(thetas, x)
                        Y_output = naive.predict(p) + 1
                        if Y_output == int(sentiment):
                            cont += 1

                        l += 1

        accuracys.append(cont/l)
