import os
from math import ceil
from time import perf_counter

import numpy as np
import pandas as pd

from MRFALKON.MRFALKONmultiResModel import MRFALKON_MultiResModel
from StreaMRAK.StreaMRAKutil.dataLoggers import DataLoggerCoef, DataLoggerMonitorScaleCover, DatabasePool, DataLoggerProgr
from StreaMRAK.falkonSolver import FALKON_Solver
from StreaMRAK.StreaMRAKconfig.loadConfigurations import loadConfigFromCSV


class DataLoader():
    def __init__(self, X_batches, y_batches):
        self.X_batches = X_batches
        self.y_batches = y_batches
        self.n_batches, _, self.embedd_dim = X_batches.shape
        _, _, self.target_dim = y_batches.shape

    def get_embedd_dim(self):
        return self.embedd_dim

    def get_target_dim(self):
        return self.target_dim

    def streamData(self, counter):
        stream = True
        if counter >= self.n_batches:
            stream = False
            return None, None, stream
        else:
            return self.X_batches[counter], self.y_batches[counter], stream



class MRFALKONconfig():
    def __init__(self, verbosity, tag='_default'):
        self.tag = tag

        # Set verbosity
        self.verbosity = verbosity

        # data loggers config
        fileNameLogger = 'loggerConfig'
        self.loggerConfig = loadConfigFromCSV(fileNameLogger, self.tag, verbosity)

        # data loggers config
        fileNameStreaMRAKmain = 'MRFALKONmainConfig'
        self.MRFALKONmainConfig = loadConfigFromCSV(fileNameStreaMRAKmain, self.tag, verbosity)

        # FALKON solver config
        fileNameFalkon = 'falkonConfig'
        self.falkon_conf = loadConfigFromCSV(fileNameFalkon, self.tag, verbosity)

    def store_config(self, directory):
        MRFALKONmainConfig_df = pd.DataFrame(self.MRFALKONmainConfig, index=[0]).transpose()
        loggerConfig_df = pd.DataFrame(self.loggerConfig, index=[0]).transpose()
        falkon_conf_df = pd.DataFrame(self.falkon_conf, index=[0]).transpose()

        filename = 'MRFALKONmainConfig'
        MRFALKONmainConfig_df.to_csv(os.path.join(directory, filename))

        filename = 'loggerConfig'
        loggerConfig_df.to_csv(os.path.join(directory, filename))

        filename = 'falkon_conf'
        falkon_conf_df.to_csv(os.path.join(directory, filename))
        return

class MRFALKON_Master():
    def __init__(self, initRadius, lm, mrfalkonconfig, ExpName):
        """
        :param initPoint: 1D np array of size d (size of embedd space), containing coord. of init. point
        :param initRadius: A float value, corresponding to an estimated span of the data
        :param initPath: an empty tuple ()
        :param streaMRAKconfig: a class object of type StreaMRAKconfig containing configuration dictionaries
        """

        self.loop = mrfalkonconfig.MRFALKONmainConfig['loop']

        self.init_scale = initRadius
        self.lm = lm

        self.newlySufficientlyCovered_lvl = False  # Only for visualizing training progress in double pendulum experiment

        # data loggers objects
        self.databasePool = DatabasePool(ExpName)
        self.dataLoggerCoef = DataLoggerCoef(mrfalkonconfig.loggerConfig, self.databasePool)
        self.dataLoggerProgr = DataLoggerProgr(ExpName)

        # FALKON solver object
        self.falkon_solver = FALKON_Solver(mrfalkonconfig.falkon_conf)

        # Initialize the multi-resolution model object
        self.multiResModel = MRFALKON_MultiResModel(self.falkon_solver, self.dataLoggerCoef, self.lm,
                                                          self.init_scale)

        self.lowlim_ntrp = mrfalkonconfig.MRFALKONmainConfig['lowlim_ntrp']

        self.interval_length_conv_est = int(mrfalkonconfig.MRFALKONmainConfig['interval_length_conv_est'])

        self.prev_lvl = 0
        self.curr_lvl = 0
        self.summary = {}
        self.counter = {}
        self.interval_counter = 0
        self.track_counter = 0
        self.finishStream = False

    def learnFromStream(self, dataLoader, ExpName, logg=False, maxLvl=None):
        """
        :param dataLoader: A class which returns:
            :param x: numpy.ndarray with shape = (xn, xd). Where xn is the number of training points
                    and xd is the embedding dimension of the data
            :param y: numpy.ndarray with shape =(yn, yd). Where yn is the number of training points
                    and yd is the dimension of the target variable.
            :param stream: Boolean flag, indicating whether to continue streaming data
        :param step: Step in the stream
        :param saveProgress: If True then progress logg is saved
        :return:
        """
        self.maxLevel = maxLvl
        self.logg = logg
        self.ExpName = ExpName

        self.embedding_dim = dataLoader.get_embedd_dim()
        self.target_dim = dataLoader.get_target_dim()

        self.lvl_waiting_for_fit = []
        self.fitted_lvls = []

        self.n_trp_lvl = {}

        self.batch_size = 30000

        self.sampleCounter = 0
        self.collectionCounter = 0

        self.start_total = perf_counter()
        batch_nr = 0

        self.init_summary(lvl=0)
        self.init_new_lvl_LaplacianPyramid(lvl=0)

        self.stream = True
        while True:
            X_batch, y_batch, self.stream = dataLoader.streamData(batch_nr)

            if self.maxLevel == None:
                pass
            elif self.curr_lvl > self.maxLevel:
                print(f"Max level {self.maxLevel}, is reached")
                self.stream = False

            batch_nr += 1
            print(f'Main Batch nr: {batch_nr}')

            if self.stream == False:
                print("Terminate streaming")
                self.stop_total = perf_counter()
                break

            for x, y in zip(X_batch, y_batch):
                x, y = np.array([x]), np.array([y])
                self.sampleCounter += 1
                if self.sampleCounter % 1000 == 0:
                    print("counter: ", self.sampleCounter)
                self.summary[f'lvl{self.curr_lvl}']['NumExposedPoints'] += 1
                self.collectTrainingData(x, y)

    def collectTrainingData(self, xtr, ytr):
        lvl = self.curr_lvl
        if lvl == 4:
            dd = 3

        if self.interval_counter == 0:
            self.xtr = xtr
            self.ytr = ytr
        else:
            self.xtr = np.concatenate((self.xtr, xtr), axis=0)
            self.ytr = np.concatenate((self.ytr, ytr), axis=0)

        self.collectionCounter += 1
        self.interval_counter += 1

        if self.interval_counter > self.interval_length_conv_est:
            self.update_matrices_at_lvl(self.xtr, self.ytr, lvl)
            self.interval_counter = 0
            print(f"lvl {lvl}: Time untill update matrices: ", perf_counter()-self.start_total)### added debug

        n_trp, d = xtr.shape
        self.n_trp_lvl[f'lvl{lvl}'] += n_trp
        self.lower_thr_trp_reached = (self.n_trp_lvl[f'lvl{lvl}'] > self.lowlim_ntrp)

        if self.lower_thr_trp_reached:
            print(f"lvl {lvl}:  Time until collect: ", perf_counter() - self.start_total) ### Added debug

            print(" ")
            print(f"Lvl {lvl} is ready")
            print(f"Number of training points / lower limit on training points ")
            print(f"{self.n_trp_lvl[f'lvl{lvl}']} / {self.lowlim_ntrp}")

            self.update_LaplacianPyramid(lvl)
            print(f"lvl {lvl}: Time until update LaplacianPyramid: ", perf_counter()-self.start_total)### added debug

            self.curr_lvl = self.curr_lvl + 1
            self.init_summary(self.curr_lvl)
            self.init_new_lvl_LaplacianPyramid(self.curr_lvl)

            self.lower_thr_trp_reached = False

        else:
            if self.collectionCounter % 10000 == 0:
                print(f"Waiting for training points, Num. training points at lvl "
                        f"{lvl} is {self.n_trp_lvl[f'lvl{lvl}']} < {self.lowlim_ntrp}")
        return


    # Fit to get coefficients
    def update_LaplacianPyramid(self, lvl):
        n_trp = self.n_trp_lvl[f'lvl{lvl}']

        print(f"num tr points when fit at lvl {lvl} is: ", n_trp)
        print(" ")

        self.multiResModel.fit_at_lvl(lvl, n_trp)
        self.curr_lvl = lvl

        self.fitted_lvls.append(lvl)

        self.summary[f'lvl{lvl}']['TimeUntilLvlReady'] = perf_counter() - self.start_total
        self.summary[f'lvl{lvl}']['NumTrPoints'] = n_trp

        if self.logg == True:
            coef = self.dataLoggerCoef.select_coef_at_level(lvl)
            self.dataLoggerProgr.logg_lm(self.lm, lvl, self.ExpName)
            self.dataLoggerProgr.logg_coef(coef, lvl, self.ExpName)
        return

    def update_matrices_at_lvl(self, xtr, ytr, lvl):
        # Update matrices at level
        n_trp, d = xtr.shape
        self.update_KnmTKnm_and_Zm_at_lvl(n_trp, xtr, ytr, lvl)
        return

    def update_KnmTKnm_and_Zm_at_lvl(self, n_trp, xtr, ytr, lvl):
        start_update_mat = perf_counter()
        d = ceil(n_trp/self.batch_size)
        xtr_batches = np.array_split(xtr, d)
        ytr_batches = np.array_split(ytr, d)
        batch_nr = 1

        if d > 1:
            print(f"Consider lvl {lvl}")

        for x, y in zip(xtr_batches, ytr_batches):
            if d > 1:
                print(f"Update in batches, batch nr: {batch_nr} of {d}")
            batch_nr += 1
            self.multiResModel.update_KnmTKnm_and_Zm_at_lvl(x, y, lvl)  # Add x,y_res as tr. data to prev. level
        stop_update_mat = perf_counter()

        if d > 1:
            print(f"Time to update matrices: {stop_update_mat - start_update_mat}")


    def init_new_lvl_LaplacianPyramid(self, lvl):
        self.multiResModel.init_new_lvl_in_LP(lvl, self.target_dim)

        self.n_trp_lvl[f'lvl{lvl}'] = 0
        self.lower_thr_trp_reached = False
        self.thr_trp_reached = False
        return

    def predict(self, X, max_lvl=None):
        """
        Make a prediction y=f(X)
        :param X: Data points, ndarray with shape = (number of points, ambient_dim)
        :return: Prediction y
        """
        return self.multiResModel.predict(X, max_lvl)

    def store_model_summary(self, path, mse_ts_list, tag):
        dataDir = os.path.join(os.getcwd(), path)
        summary_path = os.path.join(dataDir, f'summary_{tag}.csv')

        dictionary = {}
        dictionary['lvl'] = []
        dictionary['NumLandmarks'] = []
        dictionary['NumTrPoints'] = []
        dictionary['NumExposedPoints'] = []
        dictionary['TimeUntilLvlReady'] = []
        dictionary['TimeTrainTotal'] = []
        dictionary['MSEtest'] = []

        for lvl in self.multiResModel.list_of_fitted_levels:
            dictionary['lvl'].append(lvl)
            dictionary['NumLandmarks'].append(self.summary[f'lvl{lvl}']['NumLandmarks'])
            dictionary['NumTrPoints'].append(self.summary[f'lvl{lvl}']['NumTrPoints'])
            dictionary['NumExposedPoints'].append(self.summary[f'lvl{lvl}']['NumExposedPoints'])
            dictionary['TimeUntilLvlReady'].append(self.summary[f'lvl{lvl}']['TimeUntilLvlReady'])
            dictionary['TimeTrainTotal'].append(self.stop_total - self.start_total)
            dictionary['MSEtest'].append(mse_ts_list[lvl])
        df = pd.DataFrame(data=dictionary)
        df.to_csv(summary_path, sep=',', index=False)

    def init_summary(self, lvl):
        self.summary[f'lvl{lvl}'] = {}
        self.summary[f'lvl{lvl}']['NumTrPoints'] = 0

        n_lm, _ = self.lm.shape
        self.summary[f'lvl{lvl}']['NumLandmarks'] = n_lm
        self.summary[f'lvl{lvl}']['NumExposedPoints'] = 0
        self.summary[f'lvl{lvl}']['TimeUntilLvlReady'] = 0
        return