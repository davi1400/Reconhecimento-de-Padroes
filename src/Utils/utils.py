from numpy import array, zeros, where
from pandas import read_csv
from pathlib import Path


def normalize(X):
    return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))


def accuracy():
    pass


def heaveside(y):
    for i in range(len(y)):
        if y[i] > 0:
            y[i] = 1
        else:
            y[i] = 0
    return y


def get_project_root():
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.parent)


def get_data(name, type=None):
    path = get_project_root() + "/src/DataSets/" + name
    if type == 'csv':
        data = read_csv(path, header=None)

    return data


def get_inputs(data, Max_columns):
    inputs = data.iloc[:, 0:Max_columns]
    X = array(inputs).astype(float)
    return X


def get_outputs(data, Max_columns):
    outputs = data.iloc[:, Max_columns:]
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
    for label in classes.to_numpy():
        Y[count] = set_classes[label[0]]
        count += 1
    return Y


if __name__ == '__main__':
    data = get_data("Iris", type='csv')
    output = get_outputs(data, 4)
    Y = string_to_number_class(output)
