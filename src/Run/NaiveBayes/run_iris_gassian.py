from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from src.Algorithms.Supervised.NaiveBayes import NaiveBayes
from numpy import mean

if __name__ == '__main__':
    iris = load_iris()
    naive = NaiveBayes(type="gaussian")
    accuracys = []

    for realization in range(20):
        X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.2)
        Matrix_Mean, Variance_Matrix, thetas_c = naive.train(X_train, Y_train)
        accuracy = naive.test(x_test=X_test, y_test=Y_test, Matrix_Mean=Matrix_Mean, Variance_Matrix=Variance_Matrix, thetas_c=thetas_c)
        accuracys.append(accuracy)

    print(mean(accuracys))