

class CrossValidation:
    def __init__(self, X, Y, Grid, algorithm):
        self.X = X
        self.Y = Y
        self.Grid = Grid
        self.algorithm = algorithm
        self.accuracys = []

        if Grid.shape[0] != 1:
            self.flag = True
        else:
            self.flag = False


    def grid_search(self):
        if flag:
            # the shape is more tham one
            pass
        else:
            # K-fold with k = 10
            for n in self.Grid:
                L = int(self.X.shape[0] / K)
                X_trainVal = (np.c_[X_train[:L * esimo - L, :].T, X_train[esimo * L:, :].T]).T
                X_testVal = (X_train[L * esimo - L:esimo * L, :])
                Y_trainVal = (np.c_[Y_train[:L * esimo - L, :].T, Y_train[esimo * L:, :].T]).T
                Y_testVal = (Y_train[L * esimo - L:esimo * L, :])


