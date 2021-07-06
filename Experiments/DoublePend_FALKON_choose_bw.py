import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from time import time
from pathlib import Path

def load_ts_data(path):
    # Load test data
    fileName = os.path.join(path, 'state_ts.npy')
    state_ts = np.load(fileName)

    fileName = os.path.join(path, 'cm_ts.npy')
    cm_ts = np.load(fileName)

    fileName = os.path.join(path, 'poss_ts.npy')
    poss_ts = np.load(fileName)
    return state_ts, cm_ts, poss_ts

def load_tr_data(path, recording_length, num_batches, num_tr_batches, num_val_batches, forecastStep):
    # Load training data
    fileName = os.path.join(path, 'state_tr.npy')
    state_tr = np.load(fileName)

    state_tr = state_tr[:, :recording_length, :]
    state_tr_bhs = np.array(np.split(state_tr, num_batches))
    state_tr_bhs = state_tr_bhs[:, :, :-forecastStep, :]

    # Select some of the training data batches as validation data
    state_tr_bhs, state_val_bhs = train_test_split(state_tr_bhs, train_size=0.75, test_size=0.25)
    state_tr_bhs = state_tr_bhs[0:num_tr_batches]
    state_val_bhs = state_val_bhs[0:num_val_batches]

    # Select landmarks from the training batches
    potential_lm = state_tr_bhs.reshape(-1, state_tr_bhs.shape[-1])
    n_lm = 10*int(np.sqrt(potential_lm.shape[0]))
    lm_idx = np.random.randint(0, potential_lm.shape[0], size=n_lm)
    landmarks = potential_lm[lm_idx, :]

    #state_tr_bhs = state_tr_bhs[:5]
    return state_tr_bhs, state_val_bhs, landmarks

# Load falkon solver
from StreaMRAK.StreaMRAKconfig.loadConfigurations import StreaMRAKconfig
from StreaMRAK.falkonSolver import FALKON_Solver
verbosity = 1
streaMRAKconfig = StreaMRAKconfig(verbosity)
falkonSolver = FALKON_Solver(streaMRAKconfig.falkon_conf)

if __name__ == '__main__':
    ####################################################################
    pendulum_energies = ['DoublePendLowEnergy', 'DoublePendHighEnergy']
    pendulum_energy = pendulum_energies[1]
    ExperimentName = 'Dp_HighE_FALKON_choose_bw'
    config_tag = '_doublePend'
    recording_length = 500
    num_batches = 200
    num_tr_batches = 50
    num_val_batches = 10

    forecastStep = 10
    verbosity = 1

    n_bw_points = 20
    #####################################################################

    # Load data
    path = os.path.join('Datasets', pendulum_energy)
    state_tr_bhs, state_val_bhs, landmarks = load_tr_data(path, recording_length, num_batches, num_tr_batches, num_val_batches, forecastStep)

    n_lm, d = landmarks.shape
    target_dim = 4
    Kmm = np.zeros((n_lm, n_lm))
    KnmTKnm = np.zeros((n_lm, n_lm))
    Zm = np.zeros((n_lm, target_dim))

    # Grid search to determine the optimal bandwidth of the kernel
    bw_grid = np.logspace(start=2, stop=-2, num=n_bw_points)
    forecastStep = 10
    mse_list = []
    tic = time()
    for bw in bw_grid:
        n_tr = 0

        print("Bandwidth: ", bw)
        i = 0
        for state_tr in state_tr_bhs:
            print("i: ", i)
            i += 1
            y_tr = state_tr[:, forecastStep:, :]
            state_tr = state_tr[:, :-forecastStep, :]

            state_tr = state_tr.reshape(-1, state_tr.shape[-1])
            print("state_tr shape: ", state_tr.shape)
            y_tr = y_tr.reshape(-1, y_tr.shape[-1])
            n_tr = n_tr + state_tr.shape[0]
            print("n_tr: ", n_tr)

            KnmTKnm_tmp = falkonSolver.calcKnmTKnm(state_tr, landmarks, bw)
            KnmTKnm = KnmTKnm + KnmTKnm_tmp

            Zm_tmp = falkonSolver.calcZm(state_tr, y_tr, landmarks, bw)
            Zm = Zm + Zm_tmp
        Zm = (1 / n_tr) * Zm
        Kmm = falkonSolver.calcKmm(landmarks, bw)
        alpha = falkonSolver.fit_at_scale(Kmm, KnmTKnm, Zm, n_tr)
        mse_temp = []
        j=0
        for state_val in state_val_bhs:
            print("j: ", j)
            j += 1
            y_val = state_val[:, forecastStep:, :]
            state_val = state_val[:, :-forecastStep, :]

            state_val = state_val.reshape(-1, state_val.shape[-1])
            y_val = y_val.reshape(-1, y_val.shape[-1])

            y_val_pred = falkonSolver.predict_at_scale(state_val, landmarks, alpha, bw)
            mse_temp.append(mean_squared_error(y_val, y_val_pred))
            print("MSE: ", mse_temp)
        mse_list.append(np.average(mse_temp))
    opt_bw = bw_grid[mse_list.index(min(mse_list))]
    toc = time()
    time_elapse = toc-tic

    print("Bandwidth grid: ", bw_grid)
    print("MSE list ", mse_list)
    print("Optimal bandwidth: ", opt_bw)

    # Save data to file
    directory = os.path.join(os.getcwd(), 'LoggModel', ExperimentName)
    Path(directory).mkdir(parents=True, exist_ok=True)

    fileName_bw = 'bw_grid_largeE'
    path_bw = os.path.join(directory, fileName_bw)
    np.savetxt(path_bw, bw_grid)

    fileName_mse = 'mse_bw_grid_largeE'
    path_mse = os.path.join(directory, fileName_mse)
    np.savetxt(path_mse, mse_list)

    fileName_timeElapse = 'bw_time_elapse'
    path_telapse = os.path.join(directory, fileName_timeElapse)
    np.savetxt(path_telapse, np.array([time_elapse]))

    fileName_optBw = 'opt_bw'
    path_opt = os.path.join(directory, fileName_optBw)
    np.savetxt(path_opt, np.array([opt_bw]))

