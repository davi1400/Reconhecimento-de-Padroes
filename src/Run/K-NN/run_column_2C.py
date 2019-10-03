"""
    Run the column data set with K-NN algorithm
"""
import matplotlib.pyplot as plt
from src.Algorithms.Supervised.knearestneighbours import KNN
from numpy import array, arange, mean, hstack
from pandas import DataFrame
from src.Utils.utils import get_data
from src.Utils.CrossValidate import CrossValidation


if __name__ == '__main__':
    accuracys = []
    all_k = []
    data = get_data("column_2C_weka.arff", type="arff")
    data = DataFrame(data[0])
    knn = KNN(data)

    for realization in range(20):
        knn.split()
        best_k = knn.validate()
        all_k.append(best_k)
        knn.split()
        Y_output, Y_test = knn.train(best_k)
        accuracys.append(knn.test(Y_output, Y_test))
        print(" # --------------------------------- #")
        print("Realization: ", realization)
        print("Best K: ", best_k)
        print("Accuracy: ", accuracys[realization])



    print("Finished")
    print("Mean accuracy: ", mean(accuracys))
    print("Best mean K: ", int(mean(all_k)))



