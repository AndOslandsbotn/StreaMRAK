import numpy as np
from time import perf_counter

class MRFALKON_MultiResModel():
    def __init__(self, falkonSolver , dataLoggerCoef, lm, init_scale):
        self.falkonSolver = falkonSolver
        self.dataLoggerCoef = dataLoggerCoef

        self.lm = lm
        self.n_lm, d = self.lm.shape
        self.init_scale = init_scale

        self.list_of_fitted_levels = []
        self.residuals = []
        self.residuals_domain = []
        self.residuals_target = []

        self.time_calc_matrices = 0
        return

    def get_residuals(self):
        return self.residuals, self.residuals_domain, self.residuals_target

    def init_new_lvl_in_LP(self, lvl, target_dim):
        """
        Initialize the matrices KnmTKnm and Zm at level = lvl
        :param lvl: Integer, level in Cover-tree and Laplacian pyramid
        :return:
        """

        self.KnmTKnm = np.zeros((self.n_lm, self.n_lm))
        self.Zm = np.zeros((self.n_lm, target_dim))


    def update_KnmTKnm_and_Zm_at_lvl(self, X, y, lvl):
        """
        Iteratively updates the training data matrices KnmTKnm and Zm.
        :param X: The training points, ndarray with shape = (n_training_samples, ambient_dim)
        :param y: The training targets, ndarray with shape = (n_training_samples, n_target_functions)
        :param level: Integer, level in Cover-tree and Laplacian pyramid
        :return: Updated version of KnmTKnm and Zm, with the new training points X, y added
        """
        if lvl > 0:
            y_res = self.calc_residual(X, y)
        else:
            y_res = y

        print(f"Level {lvl}: we add residuals")
        self.residuals.append(y_res)
        self.residuals_domain.append(X)
        self.residuals_target.append(y)

        scale = self.init_scale/(2**lvl)
        #scale = self.init_scale

        start_calc_matrices = perf_counter()
        self.KnmTKnm = self.KnmTKnm + self.falkonSolver.calcKnmTKnm(X, self.lm, scale)
        self.Zm = self.Zm + self.falkonSolver.calcZm(X, y_res, self.lm, scale)
        stop_calc_matrices = perf_counter()
        self.time_calc_matrices = self.time_calc_matrices + stop_calc_matrices-start_calc_matrices

    def fit_at_lvl(self, lvl, n_tr_lvl):
        """
        Fit a Laplacian pyramid component s^{(l)} at the specified level, using the
        landmarks from this level in the tree and the matrices defined on these landmarks
        and the nodes at level+1.
        :param level: Integer, level in Cover-tree and Laplacian pyramid
        :param n_tr_lvl: Number of training points used to form KnmTKnm and Zm
        """
        scale = self.init_scale/(2**lvl)

        start_calc_Kmm = perf_counter()
        self.Kmm = self.falkonSolver.calcKmm(self.lm, scale)
        stop_calc_Kmm = perf_counter()
        print(f"Calc Kmm at level {lvl}, TIME: {stop_calc_Kmm-start_calc_Kmm}")

        start_fit = perf_counter()
        coef = self.falkonSolver.fit_at_scale(self.Kmm, self.KnmTKnm, (1/n_tr_lvl)*self.Zm, n_tr_lvl)
        stop_fit = perf_counter()
        print(f"Fit at level {lvl}, TIME: {stop_fit-start_fit}")

        print(f"Time to calc matrices {self.time_calc_matrices}")
        self.time_calc_matrices = 0

        self.dataLoggerCoef.store_coef_at_level(lvl, scale, coef)  # Store the coef in the prediction model
        self.list_of_fitted_levels.append(lvl)
        return

    def predict(self, X, max_lvl=None):
        """ Make a prediction y = f(x) using the currently available model
        :param X: Data points, ndarray with shape = (n_training_samples, ambient_dim)
        :return: Prediction y
        """
        if len(self.list_of_fitted_levels) == 0:
            print("No fitted levels")
            return None

        if max_lvl == None:
            max_lvl = len(self.list_of_fitted_levels)
        for lvl in self.list_of_fitted_levels[:max_lvl+1]:
            scale = self.init_scale / (2 ** lvl)
            coef = self.dataLoggerCoef.select_coef_at_level(lvl)
            if lvl == 0:
                y_pred = self.falkonSolver.predict_at_scale(X, self.lm, coef, scale)
            else:
                y_pred = y_pred + self.falkonSolver.predict_at_scale(X, self.lm, coef, scale)
        return y_pred

    def calc_residual(self, X, y):
        """
        Calculates residual between y_target and prediction made by currently available model
        :param X: Data points, ndarray with shape= (n_training_samples, ambient_dim])
        :param y: Target value, ndarray with shape = (n_training_samples, n_target_functions)
        :return: y_res
        """
        return y - self.predict(X)
