from numpy import zeros, where, nan, array, mean
from random import shuffle
from src.Utils.SentimentAnalisys import SentimentAnalisys
from sklearn.model_selection import train_test_split
from src.Utils.utils import get_data
from src.Algorithms.Supervised.NaiveBayes import NaiveBayes


def pre_processing(reviews, bag, type="tr"):
    Matrix = zeros((len(reviews), len(bag)+1))
    i = 0
    for review, sentiment in reviews:
        if str(sentiment) != 'nan' and len(sentiment.split()) == 1:
            Matrix[i] = SentimentAnalisys.text_to_vector(Matrix[i], review, int(sentiment), bag)
        i += 1

    return Matrix


if __name__ == '__main__':
    data = get_data('chennai_reviews.csv', type='csv')
    bag = (SentimentAnalisys.create_vocabulary(data[2]))

    accuracys = []
    accuracys_train = []

    for realization in range(10):
        hit = 0
        reviews_train, reviews_test = train_test_split(array(data[[2, 3]])[1:], test_size=0.2)
        Matrix_train = pre_processing(reviews_train, bag)
        naive = NaiveBayes(Matrix_train, reviews_train, bag, type="bernolli")
        thetas_ic, thetas_c = naive.train()
        for row in Matrix_train:
            p = naive.test(row=row, thetas_ic=thetas_ic, thetas_c=thetas_c)
            Y_output = naive.predict(p)+1
            Y_expected = int(row[len(row)-1])
            if Y_output == Y_expected:
                hit += 1
        accuracys_train.append(hit / (Matrix_train.shape[0] * 1.0))
        print("Realization", realization, "accuracy train: ", hit / (Matrix_train.shape[0] * 1.0))

        # ----------------------------------------------------------------------------------------------------------- #
        # test
        Matrix_test = pre_processing(reviews_test, bag, type=None)
        hit = 0
        for row in Matrix_test:
            p = naive.test(row=row, thetas_ic=thetas_ic, thetas_c=thetas_c)
            Y_output = naive.predict(p)+1
            Y_expected = int(row[len(row)-1])
            if Y_output == Y_expected:
                hit += 1

        accuracys.append(hit/(Matrix_test.shape[0]*1.0))

        print("Realization", realization, "accuracy test: ", hit/(Matrix_test.shape[0]*1.0))

    print("Acuracias de teste: ", mean(accuracys))
    print("Acuraica de treinamento: ", (mean(accuracys_train)))