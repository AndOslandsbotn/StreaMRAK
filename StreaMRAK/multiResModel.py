import numpy as np

class MultiResModel():
    def __init__(self, falkonSolver , dataLoggerCoef, dataLoggerLM):
        self.falkonSolver = falkonSolver
        self.dataLoggerCoef = dataLoggerCoef
        self.dataLoggerLM = dataLoggerLM

        self.list_of_fitted_levels = []

    def init_new_lvl_in_LP(self, lvl, target_dim):
        """
        Initialize the matrices KnmTKnm and Zm at level = lvl
        :param lvl: Integer, level in Cover-tree and Laplacian pyramid
        :return:
        """
        landmarks, scale = self.dataLoggerLM.select_lm_and_scale_at_level(lvl)  # Get Landmarks
        n_lm, _ = landmarks.shape
        self.KnmTKnm = np.zeros((n_lm, n_lm))
        self.Zm = np.zeros((n_lm, target_dim))


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
        landmarks, scale = self.dataLoggerLM.select_lm_and_scale_at_level(lvl)  # Get Landmarks

        n_lm, _ = landmarks.shape
        n_tr, _ = X.shape

        self.KnmTKnm = self.KnmTKnm + self.falkonSolver.calcKnmTKnm(X, landmarks, scale)
        self.Zm = self.Zm + self.falkonSolver.calcZm(X, y_res, landmarks, scale)


    def fit_at_lvl(self, lvl, n_tr_lvl):
        """
        Fit a Laplacian pyramid component s^{(l)} at the specified level, using the
        landmarks from this level in the tree and the matrices defined on these landmarks
        and the nodes at level+1.
        :param level: Integer, level in Cover-tree and Laplacian pyramid
        :param n_tr_lvl: Number of training points used to form KnmTKnm and Zm
        """
        landmarks, scale = self.dataLoggerLM.select_lm_and_scale_at_level(lvl)  # Get Landmarks
        self.Kmm = self.falkonSolver.calcKmm(landmarks, scale)

        coef = self.falkonSolver.fit_at_scale(self.Kmm, self.KnmTKnm, (1/n_tr_lvl)*self.Zm, n_tr_lvl)

        self.dataLoggerCoef.store_coef_at_level(lvl, scale, coef)  # Store the coef in the prediction model
        self.list_of_fitted_levels.append(lvl)
        return

    def predict(self, X, max_lvl=None):
        """ Make a prediction y = f(x) using the currently available model
        :param X: Data points, ndarray with shape = (n_training_samples, ambient_dim)
        :return: Prediction y
        """
        if max_lvl == None:
            max_lvl = len(self.list_of_fitted_levels)
        for lvl in self.list_of_fitted_levels[:max_lvl+1]:
            landmarks, scale = self.dataLoggerLM.select_lm_and_scale_at_level(lvl)
            coef = self.dataLoggerCoef.select_coef_at_level(lvl)
            if lvl == 0:
                y_pred = self.falkonSolver.predict_at_scale(X, landmarks, coef, scale)
            else:
                y_pred = y_pred + self.falkonSolver.predict_at_scale(X, landmarks, coef, scale)
        return y_pred

    def calc_residual(self, X, y):
        """
        Calculates residual between y_target and prediction made by currently available model
        :param X: Data points, ndarray with shape= (n_training_samples, ambient_dim])
        :param y: Target value, ndarray with shape = (n_training_samples, n_target_functions)
        :return: y_res
        """
        return y - self.predict(X)

    def avg_sum_matrices(self, n_tr_points):
        sum_KnmTKnm = np.sum(self.KnmTKnm)
        sum_Zm = np.sum(self.Zm)
        return sum_KnmTKnm/n_tr_points, sum_Zm/n_tr_points




