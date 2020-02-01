from scipy.io import arff
from zipfile import ZipFile
from numpy import array, zeros, where, sum, tanh, exp
from pandas import read_csv, DataFrame
from pathlib import Path
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from math import sqrt
from scipy.special import expit
from seaborn import set, heatmap


def sigmoid(logist, y):
    if logist:
        # sigmoid logistica

        try:
            return expit(y)
        except Exception:
            return array(expit(y.tolist()))
    else:
        # sigmoid tangente hiperbolica
        return (1 - exp(-y))/(1 + exp(-y))


def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def get_error_rate(y_output, y_test):
    return abs(sum(y_test != y_output)) * 1.0 / len(y_test) * 1.0


def get_accuracy(y_output, y_test):
    return abs(sum(y_test == y_output)) * 1.0 / len(y_test) * 1.0


def get_confusion_matrix(y_output, y_test):
    return confusion_matrix(y_test.tolist(), y_output.tolist())


def plot_confusion_matrix(confusion_matrix):
    df_cm = DataFrame(confusion_matrix, range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1]))
    # plt.figure(figsize=(10,7))
    set(font_scale=1.4)  # for label size
    heatmap(df_cm, annot=True, annot_kws={"size": 16})  # font size

    plt.show()


def heaveside(y):
    for i in range(len(y)):
        if y[i] > 0:
            y[i] = 1
        elif y[i] < 0:
            y[i] = 0
    return y


def get_project_root():
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.parent)


def get_data(name, type=None):
    path = get_project_root() + "/src/DataSets/" + name
    if type == 'csv':
        data = read_csv(path, header=None)
    if type == 'arff':
        data, meta = arff.loadarff(path)

    return data


def get_inputs(data, Max_columns):
    if isinstance(data, DataFrame):
        inputs = data.iloc[:, 0:Max_columns]
    else:
        inputs = data[:, 0:Max_columns]
    X = array(inputs).astype(float)
    return X


def get_outputs(data, Max_columns):
    if isinstance(data, DataFrame):
        outputs = data.iloc[:, Max_columns:]
    else:
        outputs = data[:, Max_columns]
    return outputs


def string_to_number_class(classes):
    shape = classes.shape
    Y = zeros(shape)
    set_classes = {}
    count = 0
    for label in list(set(classes.to_numpy()[:, 0])):
        set_classes.update({label: count})
        count += 1

    count = 0
    all_classes = classes.to_numpy()
    for label in all_classes:
        Y[count] = set_classes[label[0]]
        count += 1
    return Y


# def histogram_plot(data, Y, X, N_attributs, feature_names, number_classes=3):
#
#     if number_classes > 3:
#         # TODO
#         pass
#
#     else:
#         fig, axes = plt.subplots(nrows=2, ncols=2)
#         colors = ['blue', 'red', 'green']
#         classes = data.iloc[:, N_attributs:]
#         classes = list(set(classes.to_numpy()[:, 0]))
#
#         for i, ax in enumerate(axes.flat):
#             for label, color in zip(range(len(classes)), colors):
#                 ax.hist(X[where(Y == label)[0]], label=classes[label], color=color)
#                 ax.set_xlabel(feature_names[i])
#                 ax.legend(loc='upper right')
#
#         plt.show()

def calculate_euclidian_distance(X_example, example):
    dist = 0
    for i in range(len(X_example)):
        dist += (X_example[i]-example[i])**2
    return dist


def open_zip(filepath, name=""):
    zfile = ZipFile(filepath)
    zfile.extract(name, path=get_project_root() + "/src/DataSets/")


if __name__ == '__main__':
    # data = get_data("Iris", type='csv')
    # output = get_outputs(data, 4)
    # Y = string_to_number_class(output)

    open_zip("../DataSets/vertebral_column_data.zip", "column_3C_weka.arff")
    data = get_data("column_3C_weka.arff", type="arff")
