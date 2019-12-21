import numpy as np
from src.Utils.mock import create_mock
from src.Algorithms.Supervised.Perceptron import perceptron

if __name__ == '__main__':
    number_executions = 1000
    execution_wins = {"SVM": 0,
                      "PERCEPTRON": 0}
    number_points = 10

    text_execution = "Realization {0}\n  Error rate {1}\n"

    for execution in range(number_executions):

        # ------------------------------------------------------------------------------------ #

        # Step one create the data

        verify = True

        while verify:
            data, out_of_data_points = create_mock(number_points)
            if data is None:
                data, out_of_data_points = create_mock(number_points)
            else:
                verify = False
        X = data[:, :2]
        Y = data[:, 2]

        # ------------------------------------------------------------------------------------- #

        # step two, run with perceptron algorithm

        for realization in range(1):
            p = perceptron(data, learning_rate=0.015, epochs=400, type=1, logist=False)
            p.split()
            wheights, X_test, Y_test = p.train()
            error_rate = p.test(wheights, out_of_data_points[:, :2], out_of_data_points[:, 2], inverse=True)

            print(text_execution.format(realization, error_rate))

        # -------------------------------------------------------------------------------------- #

        # step three, run with support vector machines

        for realization in range(1):
            