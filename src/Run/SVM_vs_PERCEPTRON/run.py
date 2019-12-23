import numpy as np
from src.Utils.utils import get_error_rate
from src.Utils.mock import create_mock
from src.Algorithms.Supervised.Perceptron import perceptron
from src.Algorithms.Supervised.SupportVectorMachines import svm
from numpy import array

if __name__ == '__main__':
    number_executions = 10
    execution_wins = {"SVM": 0,
                      "PERCEPTRON": 0}
    number_points = 10

    text_execution = "Realization {0}\n  Error rate {1}\n Model{2}"

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
            p = perceptron(data, learning_rate=0.1, epochs=400, type=1, logist=False)
            p.split()
            wheights, X_test, Y_test = p.train(batch=False)
            error_rate_p = p.test(wheights, data[:, :2], data[:, 2])

            print(text_execution.format(realization, error_rate_p, "perceptron"))

        # -------------------------------------------------------------------------------------- #

        # step three, run with support vector machines

        for realization in range(1):
            model = svm(data, type="HardSoft")
            best_w = model.train()

            if best_w is None:
                print("Error svm")
                execution_wins['PERCEPTRON'] += 1
                continue

            accuracy_svm = model.test(out_of_data_points[:, :2], out_of_data_points[:, 2])
            error_rate_svm = 1 - accuracy_svm

            print(text_execution.format(realization, error_rate_svm, "svm"))

            if error_rate_svm < error_rate_p:
                execution_wins['SVM'] += 1
            else:
                execution_wins['PERCEPTRON'] += 1

            print(model.get_number_support_vectors())
            # print(model.vec_sup)

    print(execution_wins)