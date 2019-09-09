from pandas import read_csv
from pathlib import Path


def get_project_root():
    """Returns project root folder."""
    return str(Path(__file__).parent.parent.parent)


def get_data(name, type=None):
    path = get_project_root() + "/src/DataSets/" + name
    if type == 'csv':
        data = read_csv(path, header=None)

    return data


def get_inputs(data, columns):
    pass


def get_outpus(data, columns):
    pass


if __name__ == '__main__':
    data = get_data("Iris", type='csv')
