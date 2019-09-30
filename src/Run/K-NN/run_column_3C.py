"""
    Run the column data set with K-NN algorithm
"""
from src.Algorithms.Supervised.knearestneighbours import KNN
from numpy import array, arange
from pandas import DataFrame
from src.Utils.utils import get_data


if __name__ == '__main__':
    all_K = arange(1, 10, 3)
    data = get_data("column_3C_weka.arff", type="arff")
    data = DataFrame(data[0])
    knn = KNN(data)
    accuracys = knn.train()



