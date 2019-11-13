from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.Utils.utils import normalize
from src.Algorithms.Supervised.LogisticRegression import LogisticRegression
from numpy import where, mean

if __name__ == '__main__':

    iris = load_iris()
    logistic = LogisticRegression(eta=0.1, ephocs=1000)
    accuracys = []
    target = iris.target.copy()
    indices = where(iris.target == 2)
    target[indices] = 1
    indices = where(iris.target == 0)
    target[indices] = -1

    for realization in range(20):
        X_train, X_test, Y_train, Y_test = train_test_split(normalize(iris.data), target.T, test_size=0.2)
        W = logistic.train(X_train, Y_train)
        accuracys.append(logistic.test(W, X_test, Y_test))

    print(mean(accuracys))