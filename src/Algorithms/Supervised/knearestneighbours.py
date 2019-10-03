from src.Utils.utils import get_inputs, get_outputs, string_to_number_class
from numpy.random import rand
from numpy import where, array
from src.Utils.utils import calculate_euclidian_distance, get_accuracy
from sklearn.model_selection import train_test_split
from src.Utils.CrossValidate import CrossValidation


# TODO
class KNN:
    def __init__(self, data=None, test_rate=0.2, features=None):
        self.X_train = []
        self.X_test = []
        self.Y_train = []
        self.Y_test = []
        if data is None:
            self.test_rate = test_rate

        self.test_rate = test_rate
        self.number_lines = data.shape[0]
        self.number_columns = data.shape[1]
        self.X = get_inputs(data, self.number_columns - 1)
        self.classes = get_outputs(data, self.number_columns - 1)
        self.Y = string_to_number_class(self.classes)


    def validate(self):
        if self.validate:
            grid = range(1, 10, 2)
            validation = CrossValidation(grid, self)
            return validation.validate()


    def split(self):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.X, self.Y,
                                                                                test_size=self.test_rate)

    def train(self, *argv):
        k = argv[0]
        return self.classify(self.X_train, self.X_test, self.Y_train, self.Y_test, k)

    def test(self, Y_output, Y_test):
        accuracy = get_accuracy(array(Y_output, dtype='int', ndmin=2).T, Y_test)
        return accuracy

    def classify(self, X_train, X_test, Y_train, Y_test, k):
        Y_output = []
        for exampe in X_test:
            distance = []
            for indice in range(len(X_train)):
                euclidian_dist = calculate_euclidian_distance(X_train[indice], exampe)
                distance.append((euclidian_dist, Y_train[indice]))

            k_nearest_distances_and_labels = sorted(distance)[:k]
            counter = {}
            for i in range(len(k_nearest_distances_and_labels)):
                label = str(int(k_nearest_distances_and_labels[i][1][0]))
                if label in counter:
                    counter[label] += 1
                else:
                    counter.update({label: 1})

            max_one = max(list(counter.values()))
            for label in counter.keys():
                if counter[label] == max_one:
                    Y_output.append(label)
                    break

        return Y_output, Y_test

    def transform_binary(self, name):
        self.Y[where(self.classes == name)] = 0
        self.Y[where(self.classes != name)] = 1


