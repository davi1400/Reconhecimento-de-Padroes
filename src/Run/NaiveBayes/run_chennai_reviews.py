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
    reviews_train, reviews_test = train_test_split(array(data[[2, 3]])[1:], test_size=0.2)
    bag = list(SentimentAnalisys.create_vocabulary(data[2]))

    for realization in range(1):
        Matrix_train = pre_processing(reviews_train, bag)
        print(Matrix_train)

    naive = NaiveBayes(Matrix_train, reviews_train, bag)
    naive.train()
