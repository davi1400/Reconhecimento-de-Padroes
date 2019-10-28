import nltk
import numpy
from enchant import Dict
from nltk.corpus import stopwords
from src.Utils.utils import get_data
from random import shuffle

nltk.download('punkt')
nltk.download('stopwords')


class SentimentAnalisys:

    @staticmethod
    def create_vocabulary(documents):
        vocabulary = set()

        for document in documents:
            if document is not numpy.nan:
                words = SentimentAnalisys.filter_stopwords(document)
                vocabulary.update(words)

        return vocabulary

    @staticmethod
    def text_to_vector(row, text, sentiment, bag_of_words, i):
        text_filtered = SentimentAnalisys.filter_stopwords(text)
        bag_of_words = numpy.array(bag_of_words, ndmin=2)
        for word in text_filtered:
            row[numpy.where(word == bag_of_words)[1][0]] += 1

        row[len(row) - 1] = sentiment
        # print(row[len(row) - 1], sentiment, i)
        return row

    @staticmethod
    def filter_stopwords(text):
        text_tokenized = set(nltk.word_tokenize(text))
        english_stop_words = set(stopwords.words('english'))
        text_filtered = text_tokenized - english_stop_words

        return text_filtered


if __name__ == '__main__':
    text = "A graphical model or probabilistic graphical model (PGM) is a probabilistic model for which a graph " \
           "expresses the conditional dependence structure between random variables. They are commonly used in " \
           "probability theory, statistics—particularly Bayesian statistics—and machine learning."

    text_filtered = SentimentAnalisys.filter_stopwords(text)
    # d = SentimentAnalisys.text_to_vector(text)

    data = get_data('chennai_reviews.csv', type='csv')
    reviews = data[2]
    vocab = list(SentimentAnalisys.create_vocabulary(reviews[1:]))
    shuffle(vocab)


