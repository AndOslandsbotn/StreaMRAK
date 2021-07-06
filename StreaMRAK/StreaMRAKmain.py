import numpy as np
from time import perf_counter
import queue
import pandas as pd
import os
from math import ceil

from StreaMRAK.falkonSolver import FALKON_Solver
from StreaMRAK.StreaMRAKutil.dataLoggers import DataLoggerLM, DataLoggerCoef, DataLoggerMonitorScaleCover, DataLoggerProgr, DatabasePool
from StreaMRAK.StreaMRAKconfig.loadConfigurations import loadConfigFromCSV
from StreaMRAK.dampedCoverTree import DampedCoverTreeMaster
from StreaMRAK.multiResModel import MultiResModel

class DataLoader():
    def __init__(self, X_batches, y_batches):
        self.X_batches = X_batches
        self.y_batches = y_batches
        self.n_batches, _, self.embedd_dim = X_batches.shape
        _, _, self.target_dim = y_batches.shape

    def get_n_batches(self):
        return self.n_batches

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


class DataLoader2():
    def __init__(self, X_batches, y_batches):
        self.X_batches = X_batches
        self.y_batches = y_batches
        self.n_batches = len(X_batches)
        _,  self.embedd_dim = X_batches[0].shape
        _, self.target_dim = y_batches[0].shape

    def get_n_batches(self):
        return self.n_batches

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
            d = self.X_batches[counter]
            return self.X_batches[counter], self.y_batches[counter], stream


class StreaMRAKconfig():
    def __init__(self, verbosity, tag='_default'):
        self.tag = tag

        # Set verbosity
        self.verbosity = verbosity

        # data loggers config
        fileNameLogger = 'loggerConfig'
        self.loggerConfig = loadConfigFromCSV(fileNameLogger, self.tag, verbosity)

        # data loggers config
        fileNameStreaMRAKmain = 'streaMRAKmainConfig'
        self.streaMRAKmainConfig = loadConfigFromCSV(fileNameStreaMRAKmain, self.tag, verbosity)

        # FALKON solver config
        fileNameFalkon = 'falkonConfig'
        self.falkon_conf = loadConfigFromCSV(fileNameFalkon, self.tag, verbosity)

        # monitorScaleCover config
        fileNameMonitorScale = 'monitorScaleCovConfig'
        self.monitor_conf = loadConfigFromCSV(fileNameMonitorScale, self.tag, verbosity)

        # CoverTree object
        fileNameCoverTree = 'coverTreeConfig'
        self.coverTree_conf = loadConfigFromCSV(fileNameCoverTree, self.tag, verbosity)

    def store_config(self, directory):
        streaMRAKmainConfig_df = pd.DataFrame(self.streaMRAKmainConfig, index=[0]).transpose()
        loggerConfig_df = pd.DataFrame(self.loggerConfig, index=[0]).transpose()
        falkon_conf_df = pd.DataFrame(self.falkon_conf, index=[0]).transpose()
        monitor_conf_df = pd.DataFrame(self.monitor_conf, index=[0]).transpose()
        coverTree_conf_df = pd.DataFrame(self.coverTree_conf, index=[0]).transpose()

        filename = 'streaMRAKmainConfig'
        streaMRAKmainConfig_df.to_csv(os.path.join(directory, filename))

        filename = 'loggerConfig'
        loggerConfig_df.to_csv(os.path.join(directory, filename))

        filename = 'falkon_conf'
        falkon_conf_df.to_csv(os.path.join(directory, filename))

        filename = 'monitor_conf'
        monitor_conf_df.to_csv(os.path.join(directory, filename))

        filename = 'coverTree_conf'
        coverTree_conf_df.to_csv(os.path.join(directory, filename))
        return

class StreaMRAKmaster():
    def __init__(self, initPoint, initTarget, initRadius, initPath, streaMRAKconfig, ExpName):
        """
        :param initPoint: 1D np array of size d (size of embedd space), containing coord. of init. point
        :param initRadius: A float value, corresponding to an estimated span of the data
        :param initPath: an empty tuple ()
        :param streaMRAKconfig: a class object of type StreaMRAKconfig containing configuration dictionaries
        """
        self.newlySufficientlyCovered_lvl = False  # Only for visualizing training progress in double pendulum experiment

        # data loggers objects
        self.databasePool = DatabasePool(ExpName)
        self.dataLoggerCoef = DataLoggerCoef(streaMRAKconfig.loggerConfig, self.databasePool)
        self.dataLoggerLM = DataLoggerLM(streaMRAKconfig.loggerConfig, self.databasePool)
        self.dataLoggerProgr = DataLoggerProgr(ExpName)

        # FALKON solver object
        self.falkon_solver = FALKON_Solver(streaMRAKconfig.falkon_conf)

        # monitorScaleCover object
        self.monitorScaleCover = DataLoggerMonitorScaleCover(streaMRAKconfig.monitor_conf)
        self.monitorScaleCover.update_n_nodes_at_lvl(lvl=0) # Add the root node

        # Initialize the multi-resolution model object
        self.multiResModel = MultiResModel(self.falkon_solver,
                                           self.dataLoggerCoef, self.dataLoggerLM)

        # Initialize the CoverTree object
        self.dampCoverTree = DampedCoverTreeMaster(initPoint, initTarget, initRadius, initPath,
                                            streaMRAKconfig.coverTree_conf['useRandomizer'],
                                            self.monitorScaleCover)
        self.dampCoverTree.root = self.dampCoverTree

        self.conv_KnmTKnm_thr = streaMRAKconfig.streaMRAKmainConfig['conv_KnmTKnm_thr']
        self.conv_Zm_thr = streaMRAKconfig.streaMRAKmainConfig['conv_Zm_thr']

        self.use_conv_est = int(streaMRAKconfig.streaMRAKmainConfig['use_conv_est'])
        self.track_length_conv_est = int(streaMRAKconfig.streaMRAKmainConfig['track_length_conv_est'])
        self.interval_length_conv_est = int(streaMRAKconfig.streaMRAKmainConfig['interval_length_conv_est'])

        self.lowlim_ntrp = streaMRAKconfig.streaMRAKmainConfig['lowlim_ntrp']
        self.uplim_ntrp_factor = streaMRAKconfig.streaMRAKmainConfig['uplim_ntrp_factor']

        self.extract_all_nodes = streaMRAKconfig.streaMRAKmainConfig['extract_all_nodes']

        self.curr_lvl = -1
        self.summary = {}
        self.counter = {}
        self.interval_counter = 0
        self.track_counter = 0
        self.finishStream = False

    def learnFromStream(self, dataLoader, ExpName, logg=False):
        """
        :param dataLoader: A class which returns:
            :param x: numpy.ndarray with shape = (xn, xd). Where xn is the number of training points
                    and xd is the embedding dimension of the data
            :param y: numpy.ndarray with shape =(yn, yd). Where yn is the number of training points
                    and yd is the dimension of the target variable.
            :param stream: Boolen flag, indicating whether to continue streaming data
        :param step: Step in the stream
        :param saveProgress: If True then progress logg is saved
        :return:
        """
        self.logg = logg
        self.ExpName = ExpName

        self.embedding_dim = dataLoader.get_embedd_dim()
        self.target_dim = dataLoader.get_target_dim()

        self.curr_included = []
        self.lvls_waiting_for_staging = []
        self.lvl_waiting_for_trData = []
        self.lvl_waiting_for_fit = []
        self.fitted_lvls = []

        self.KnmTKnm_dict = {}
        self.Zm_dict = {}

        self.n_trp_lvl = {}
        self.is_converged = False
        self.checkConv = False

        self.batch_size = 10000

        self.sampleCounter = 0
        self.collectionCounter = 0

        self.start_total = perf_counter()
        batch_nr = 0

        self.init_collection = True

        self.stream = True
        while True:
            X_batch, y_batch, self.stream = dataLoader.streamData(batch_nr)

            if self.stream == False:
                print("Stream is false")
                if len(self.lvl_waiting_for_trData) > 0:
                    print("Stream is false, lvl_waiting_for_trData: ", self.lvl_waiting_for_trData)
                    lvl = self.lvl_waiting_for_trData.pop(0)
                    print(f'Including lvl {lvl}')
                    self.lvl_waiting_for_fit.append(lvl)
                    self.update_LaplacianPyramid()
                    self.monitorScaleCover.log_at_intervals()

                if len(self.lvls_waiting_for_staging) > 0:
                    print("Stream is false, include lvls_waiting_for_staging: ", self.lvls_waiting_for_staging)
                    while len(self.lvls_waiting_for_staging) > 0:
                        lvl = self.lvls_waiting_for_staging.pop(0)
                        print(f'Including lvl {lvl}')
                        self.init_summary(lvl)
                        self.init_new_lvl_LaplacianPyramid(lvl)
                        self.extract_nodes_as_trData(lvl)
                        self.lvl_waiting_for_fit.append(lvl)
                        self.update_LaplacianPyramid()
                        self.monitorScaleCover.log_at_intervals()

                if len(self.monitorScaleCover.suff_node_cover_waiting) > 0:
                    print("Stream is false, include levels sufficiently covered but lacking enough lm", self.monitorScaleCover.suff_node_cover_waiting)
                    while len(self.monitorScaleCover.suff_node_cover_waiting) > 0:
                        lvl = self.monitorScaleCover.suff_node_cover_waiting.pop(0)
                        if self.dampCoverTree.get_numLmAtLvl(lvl) > 0:
                            self.init_summary(lvl)
                            self.extract_and_set_lm(lvl)
                            self.init_new_lvl_LaplacianPyramid(lvl)
                            self.extract_nodes_as_trData(lvl)
                            self.lvl_waiting_for_fit.append(lvl)
                            self.update_LaplacianPyramid()
                            self.monitorScaleCover.log_at_intervals()
                        else:
                            print("No landmarks at this level")

                for lvl in range(0, len(self.monitorScaleCover.coverTreeSummary[f'NumNodesWhenFinish'])):
                    if lvl >= self.monitorScaleCover.allocSize:
                        self.monitorScaleCover.update_alloc_size()
                    self.monitorScaleCover.coverTreeSummary[f'NumNodesWhenFinish'][lvl] = self.monitorScaleCover.n_nodes_at_lvl[lvl]
                    self.monitorScaleCover.coverTreeSummary[f'NumLMwhenFinish'][lvl] = self.monitorScaleCover.n_lm_at_lvl[lvl]
                self.stop_total = perf_counter()
                self.monitorScaleCover.save_lvlfilling_log(ExpName)
                break

            else:
                batch_nr += 1
                print(f'Main Batch nr: {batch_nr}')
                for x, y in zip(X_batch, y_batch):
                    x, y = np.array([x]), np.array([y])
                    self.sampleCounter += 1
                    if self.curr_lvl >= 0:
                        self.summary[f'lvl{self.curr_lvl}']['NumExposedPoints'] += 1
                    self.organizeData(x, y)
                    self.collectTrainingData(x, y)
                    self.update_LaplacianPyramid()
                    self.monitorScaleCover.log_at_intervals()
        return

    def organizeData(self, x, y):
        self.dampCoverTree.insert(x, y)

        if self.sampleCounter % 10000 == 0:
            print(f"Sample nr {self.sampleCounter}")

        if len(self.monitorScaleCover.suff_lm_cover_waiting) > 0:
            lvl = self.monitorScaleCover.suff_lm_cover_waiting.pop(0)
            self.init_summary(lvl)
            self.extract_and_set_lm(lvl)
            self.lvls_waiting_for_staging.append(lvl)
            self.curr_included.append(lvl)

        if len(self.lvls_waiting_for_staging) > 0:
            lvl = min(self.lvls_waiting_for_staging)
            if lvl == 0:
                self.init_new_lvl_LaplacianPyramid(lvl)  # Initialize KnmTKnm and Zm matrices for the first lvl
                self.extract_nodes_as_trData(lvl)

                self.lvls_waiting_for_staging.remove(lvl)
                self.lvl_waiting_for_trData.append(lvl)
            elif lvl - 1 in self.fitted_lvls:
                self.init_new_lvl_LaplacianPyramid(lvl)  # Initialize KnmTKnm and Zm matrices for the next lvl
                self.extract_nodes_as_trData(lvl)

                self.lvls_waiting_for_staging.remove(lvl)
                self.lvl_waiting_for_trData.append(lvl)
            else:
                pass
            return

    def extract_and_set_lm(self, lvl):
        lmExtract, pot_lm, scale = self.dampCoverTree.select_lm(lvl)

        numExtractLM, _ = lmExtract.shape
        numLM, _ = pot_lm.shape

        self.multiResModel.dataLoggerLM.store_lm_and_scale_at_level(lvl, scale, lmExtract)

        self.summary[f'lvl{lvl}']['NumLandmarksExtracted'] = numExtractLM
        self.summary[f'lvl{lvl}']['NumLandmarks'] = numLM
        print(f'Number of landmarks extracted {numExtractLM}')
        return

    def extract_nodes_as_trData(self, lvl):
        if self.extract_all_nodes == True:
            self.extract_nodes_as_trData1(lvl)
        else:
            self.extract_nodes_as_trData2(lvl)


    def extract_nodes_as_trData1(self, lvl):
        xtr, ytr = self.dampCoverTree.select_all_nodes()
        #xtr, ytr = self.dampCoverTree.select_nodes_as_trData(lvl+1)

        n, d = xtr.shape
        self.update_matrices_at_lvl(xtr, ytr, lvl)

        print(f"Number of nodes from lvl {lvl + 1} selected for lvl {lvl} is: {n}")
        return

    def extract_nodes_as_trData2(self, lvl):
        """Ensure minimum number of training data"""
        extract_lvl = lvl+1
        xtr, ytr = self.dampCoverTree.select_nodes_as_trData(extract_lvl)  # Select nodes at lvl 1 as tr data
        n, d = xtr.shape
        if n < self.lowlim_ntrp:
            xtr2, ytr2 = self.dampCoverTree.select_all_nodes()
            n2, d2 = xtr2.shape
            pot_idx = np.arange(n2)
            num_select = min(int(self.lowlim_ntrp+1-n), n2)
            idx = np.random.choice(pot_idx, replace=False, size=num_select)
            xtr = np.concatenate((xtr, xtr2[idx]), axis=0)
            ytr = np.concatenate((ytr, ytr2[idx]), axis=0)
        if n > self.lowlim_ntrp:
            pot_idx = np.arange(n)
            idx = np.random.choice(pot_idx, replace=False, size=int(self.lowlim_ntrp))
            xtr = xtr[idx]
            ytr = ytr[idx]
        n, d = xtr.shape
        self.update_matrices_at_lvl(xtr, ytr, lvl)
        print(f"Number of nodes from lvl {extract_lvl} selected for lvl {lvl} is: {n}")
        return

    def collectTrainingData(self, xtr, ytr):
        if len(self.lvl_waiting_for_trData) > 0:
            lvl = self.lvl_waiting_for_trData[0]
            if (lvl == 0) or (lvl - 1 in self.fitted_lvls):

                if self.checkConv == True:
                    self.collectionCounter += 1
                    self.track_counter += 1
                    self.update_matrices_at_lvl(xtr, ytr, lvl)
                    self.is_converged = self.check_conv(lvl)

                else:
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
                        self.checkConv = True

            self.lower_thr_trp_reached = (self.n_trp_lvl[f'lvl{lvl}'] > self.lowlim_ntrp)
            self.upper_thr_trp_reached = (self.n_trp_lvl[f'lvl{lvl}'] > self.uplim_ntrp_factor)

            if (self.is_converged and self.lower_thr_trp_reached) \
                    or (self.upper_thr_trp_reached and self.lower_thr_trp_reached):
                print(" ")
                print(f"Lvl {lvl} is ready")
                print(f"convergence: {self.is_converged}")
                print(f"Current number of training points "
                      f"/ Upper limit on training points "
                      f"/ Lower limit on training points ")
                print(f"{self.n_trp_lvl[f'lvl{lvl}']} "
                      f"/ {self.uplim_ntrp_factor} "
                      f"/ {self.lowlim_ntrp}")
                print(" ")

                self.lvl_waiting_for_trData.remove(lvl)
                self.lvl_waiting_for_fit.append(lvl)
                self.is_converged = False
                self.lower_thr_trp_reached = False
                self.upper_thr_trp_reached = False


    def update_matrices_at_lvl(self, xtr, ytr, lvl):
        if lvl == 9:
            dstop=3
        # Update matrices at level
        n_trp, d = xtr.shape
        self.update_KnmTKnm_and_Zm_at_lvl(n_trp, xtr, ytr, lvl)
        self.n_trp_lvl[f'lvl{lvl}'] += n_trp

        self.KnmTKnm_dict[f'lvl{lvl}'][0] = self.KnmTKnm_dict[f'lvl{lvl}'][1]
        self.Zm_dict[f'lvl{lvl}'][0] = self.Zm_dict[f'lvl{lvl}'][1]

        self.KnmTKnm_dict[f'lvl{lvl}'][1] = self.multiResModel.KnmTKnm
        self.Zm_dict[f'lvl{lvl}'][1] = self.multiResModel.Zm
        return

    def check_conv(self, lvl):
        is_converged = False
        KnmTKnm_infNorm, Zm_2Norm = self.calc_norms(lvl)
        is_satisfied = self.check_conv_condition(KnmTKnm_infNorm, Zm_2Norm)

        if is_satisfied and self.track_counter < self.track_length_conv_est:
            print(" ")
            print("KnmTKnm_infNorm: ", KnmTKnm_infNorm)
            print("Zm_2Norm: ", Zm_2Norm)
            print(f"Convergence at lvl {lvl} track step {self.track_counter} is satisfied, track step: {self.track_counter}/{self.track_length_conv_est}")
            print(" ")
            pass

        elif is_satisfied == False:
            print(" ")
            print(f"Stop and postpone convergence test at lvl {lvl}")
            print(f"Number of samples collected at this lvl: {self.n_trp_lvl[f'lvl{lvl}']}")
            print("KnmTKnm_infNorm: ", KnmTKnm_infNorm)
            print("Zm_2Norm: ", Zm_2Norm)
            print(" ")
            self.checkConv = False  # We stop checking convergence
            is_converged = False
            self.track_counter = 0

        elif is_satisfied and self.track_counter >= self.track_length_conv_est:
            print(" ")
            print("Convergence is completed")
            print("KnmTKnm_infNorm: ", KnmTKnm_infNorm)
            print("Zm_2Norm: ", Zm_2Norm)
            print(" ")
            self.checkConv = False
            is_converged = True
            self.track_counter = 0

        return is_converged


    def calc_norms(self, lvl):
        # For comparison
        acqNr = self.n_trp_lvl[f'lvl{lvl}']
        print(f"AT level {lvl} we have now acqNr {acqNr}")
        KnmTKnm_diff = self.KnmTKnm_dict[f'lvl{lvl}'][0] / (acqNr-1) - self.KnmTKnm_dict[f'lvl{lvl}'][1] / (acqNr)
        Zm_diff = self.Zm_dict[f'lvl{lvl}'][0] / (acqNr-1) - self.Zm_dict[f'lvl{lvl}'][1] / (acqNr)

        KnmTKnm_infNorm = np.linalg.norm(KnmTKnm_diff, ord=float('inf'))
        Zm_2Norm = np.linalg.norm(Zm_diff, ord=2)
        return KnmTKnm_infNorm, Zm_2Norm

    def check_conv_condition(self, KnmTKnm_infNorm, Zm_2Norm):
        """
        Check change in KnmTKnm and Zm at lvl, for new training points. I.e. if ||KnmTKnm(n) - KnmTKnm(n+1)||_inf
        and ||Zm(n)-Zm(n+1)||_2 are less than threshold
        :param KnmTKnm_infNorm: Inf norm ||KnmTKnm(n) - KnmTKnm(n+1)||_inf
        :param Zm_2Norm: 2 norm ||Zm(n)-Zm(n+1)||_2
        :return: boolean
        """

        print("Check conv condition:")
        print(f"KnmTKnm_infNorm: {KnmTKnm_infNorm}")

        print("Check conv condition:")
        print(f"Zm_2Norm: {Zm_2Norm}")
        if KnmTKnm_infNorm > self.conv_KnmTKnm_thr:
            cond1 = False
        else:
            cond1 = True

        if Zm_2Norm > self.conv_Zm_thr:
            cond2 = False
        else:
            cond2 = True

        if cond1 and cond2:
            is_satisfied = True
        else:
            is_satisfied = False

        return is_satisfied

    def update_LaplacianPyramid(self):
        while len(self.lvl_waiting_for_fit) > 0:
            lvl = self.lvl_waiting_for_fit[0]

            # Fit to get coefficients
            n_trp = self.n_trp_lvl[f'lvl{lvl}']

            print(f"num tr points when fit at lvl {lvl} is: ", n_trp)
            print(" ")
            start_fitTime = perf_counter()
            self.multiResModel.fit_at_lvl(lvl, n_trp)
            self.curr_lvl = lvl

            self.lvl_waiting_for_fit.remove(lvl)
            self.fitted_lvls.append(lvl)
            stop_fitTime = perf_counter()

            print(f"time until lvl{lvl} is ready", perf_counter() - self.start_total)
            self.summary[f'lvl{lvl}']['TimeUntilLvlReady'] = perf_counter() - self.start_total
            self.summary[f'lvl{lvl}']['NumTrPoints'] = n_trp

            if self.logg == True:
                lm, _ = self.dataLoggerLM.select_lm_and_scale_at_level(lvl)
                coef = self.dataLoggerCoef.select_coef_at_level(lvl)
                self.dataLoggerProgr.logg_lm(lm, lvl, self.ExpName)
                self.dataLoggerProgr.logg_coef(coef, lvl, self.ExpName)
        return

    def init_new_lvl_LaplacianPyramid(self, lvl):
        self.multiResModel.init_new_lvl_in_LP(lvl, self.target_dim)

        self.n_trp_lvl[f'lvl{lvl}'] = 0
        self.is_converged = False
        self.lower_thr_trp_reached = False
        self.upper_thr_trp_reached = False

        self.KnmTKnm_dict[f'lvl{lvl}'] = [0, 0]
        self.Zm_dict[f'lvl{lvl}'] = [0, 0]
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
        self.summary[f'lvl{lvl}']['TimeUpdateMatrices'] = stop_update_mat - start_update_mat

        if d > 1:
            print(f"Time to update matrices: {stop_update_mat - start_update_mat}")

    def predict(self, X, max_lvl=None):
        """
        Make a prediction y=f(X)
        :param X: Data points, ndarray with shape = (number of points, ambient_dim)
        :return: Prediction y
        """
        return self.multiResModel.predict(X, max_lvl)

    def store_model_summary(self, path, mse_ts_list, tag):
        dataDir = os.path.join(os.getcwd(), path)
        summary_path = os.path.join(dataDir, f'StreaMRAK_summary_{tag}.csv')

        dictionary = {}
        dictionary['lvl'] = []
        dictionary['NumLandmarksExtracted'] = []
        dictionary['NumLandmarks'] = []
        dictionary['NumTrPoints'] = []
        dictionary['TimeUntilLvlReady'] = []
        dictionary['TimeTrainTotal'] = []
        dictionary['MSEtest'] = []

        for lvl in self.multiResModel.list_of_fitted_levels:
            dictionary['lvl'].append(lvl)
            dictionary['NumLandmarksExtracted'].append(self.summary[f'lvl{lvl}']['NumLandmarksExtracted'])
            dictionary['NumLandmarks'].append(self.summary[f'lvl{lvl}']['NumLandmarks'])
            dictionary['NumTrPoints'].append(self.summary[f'lvl{lvl}']['NumTrPoints'])
            dictionary['TimeUntilLvlReady'].append(self.summary[f'lvl{lvl}']['TimeUntilLvlReady'])
            dictionary['TimeTrainTotal'].append(self.stop_total - self.start_total)
            dictionary['MSEtest'].append(mse_ts_list[lvl])
        df = pd.DataFrame(data=dictionary)
        df.to_csv(summary_path, sep=',', index=False)

        # Store cover tree info
        ct_summary_path = os.path.join(dataDir, f'ct_summary_{tag}.csv')
        df_ct_summary = pd.DataFrame(data=self.monitorScaleCover.coverTreeSummary)
        df_ct_summary.to_csv(ct_summary_path, sep=',', index=False)

    def init_summary(self, lvl):
        self.summary[f'lvl{lvl}'] = {}
        self.summary[f'lvl{lvl}']['NumTrPoints'] = 0
        self.summary[f'lvl{lvl}']['NumLandmarksExtracted'] = 0
        self.summary[f'lvl{lvl}']['NumLandmarks'] = 0
        self.summary[f'lvl{lvl}']['NumExposedPoints'] = 0
        self.summary[f'lvl{lvl }']['TimeUpdateMatrices'] = 0
        self.summary[f'lvl{lvl}']['TotTimeCaclMatrixNorms'] = 0
        self.summary[f'lvl{lvl}']['TimeUntilLvlReady'] = 0
        return