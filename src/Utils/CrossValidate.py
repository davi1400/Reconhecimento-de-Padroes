from sklearn.model_selection import KFold
from numpy import argmax, array, mean
import numpy as np
import matplotlib.pyplot as plt


class CrossValidation:
    def __init__(self, Grid=None, algorithm=None):
        if Grid and algorithm:
            self.train_data = algorithm.X_train
            self.test_data = algorithm.Y_train
            self.Grid = Grid
            self.algorithm = algorithm
            self.accuracys = []
        self.kf = KFold(n_splits=10)

        if isinstance(Grid, list):
            self.flag = True
        else:
            self.flag = False

    def validate(self):
        best_one = self.grid_search()
        return best_one

    def grid_search(self):
        percents = []
        if self.flag:
            # the shape is more tham one
            pass
        else:
            # K-fold with k = 10
            for example in self.Grid:
                for train_index, test_index in self.kf.split(self.train_data):
                    # print("TRAIN:", train_index, "TEST:", test_index)
                    X_train, X_test = self.train_data[train_index], self.train_data[test_index]
                    y_train, y_test = self.test_data[train_index], self.test_data[test_index]

                    self.algorithm.X_train = X_train
                    self.algorithm.X_test = X_test
                    self.algorithm.Y_train = y_train
                    self.algorithm.Y_test = y_test

                    Y_output, Y_test = self.algorithm.train(example)
                    accuracy = self.algorithm.test(Y_output, Y_test)
                    percents.append(accuracy)
                self.accuracys.append(mean(percents))
                percents = []
            max_indice = argmax(array(self.accuracys))

            return self.Grid[max_indice]

    def plot_cv_indices(self, X, y, group, ax, n_splits, lw=10):
        """Create a sample plot for indices of a cross-validation object."""
        np.random.seed(1338)
        cmap_data = plt.cm.Paired
        cmap_cv = plt.cm.coolwarm

        # Generate the training/testing visualizations for each CV split
        for ii, (tr, tt) in enumerate(self.kf.split(X=X, y=y, groups=group)):
            # Fill in indices with the training/test groups
            indices = np.array([np.nan] * len(X))
            indices[tt] = 1
            indices[tr] = 0

            # Visualize the results
            ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                       c=indices, marker='_', lw=lw, cmap=cmap_cv,
                       vmin=-.2, vmax=1.2)

        # Plot the data classes and groups at the end
        # ax.scatter(range(len(X)), [ii + 1.5] * len(X),
        #            c=y, marker='_', lw=lw, cmap=cmap_data)

        ax.scatter(range(len(X)), [ii + 1.5] * len(X),
                   c=group, marker='_', lw=lw, cmap=cmap_data)

        # Formatting
        yticklabels = list(range(n_splits)) + ['group']
        ax.set(yticks=np.arange(n_splits + 1) + .5, yticklabels=yticklabels,
               xlabel='Indices do conjunto de dados', ylabel="Iteração",
               ylim=[n_splits + 2.2, -.2], xlim=[0, 310])
        ax.set_title('{}'.format(type(self.kf).__name__), fontsize=15)
        return ax

