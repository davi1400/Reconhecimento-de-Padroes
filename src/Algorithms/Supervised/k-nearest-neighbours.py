from src.Utils.utils import get_inputs, get_outputs, string_to_number_class
from numpy.random import rand
from numpy import where
from src.Utils.utils import calculate_euclidian_distance


class KNN:
    def __init__(self, data):
        self.number_lines = data.shape[0]
        self.number_columns = data.shape[1]
        self.X = get_inputs(data, self.number_columns - 1)
        self.classes = get_outputs(data, self.number_columns - 1)
        self.Y = string_to_number_class(self.classes)

    def classify(self, test_data, k):
        label_examples = []
        for exampe in test_data:
            distance = []
            for indice in range(len(self.X)):
                euclidian_dist = calculate_euclidian_distance(self.X[indice], exampe)
                distance.append((euclidian_dist, self.Y[indice]))

            k_nearest_distances_and_labels = sorted(distance)[:k]
            counter = {}
            for dist, label in k_nearest_distances_and_labels:
                if counter[label]:
                    counter[label] += 1
                else:
                    counter.update({label: 1})

            max_one = max(list(counter.values()))
            for label in counter.keys():
                if counter[label] == max_one:
                    label_examples.append(label)
                    break

        return label_examples
