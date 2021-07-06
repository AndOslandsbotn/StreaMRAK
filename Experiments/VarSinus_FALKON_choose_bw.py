import numpy as np
import os
from sklearn.metrics import mean_squared_error
from pathlib import Path
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from time import perf_counter
import matplotlib

def loop_data(x, y, n):
    """
    :param x: domain
    :param y: target
    :param n: number of times to loop
    :return:
    """
    i = 1
    while i < n:
        x = np.concatenate((x, x), axis=0)
        y = np.concatenate((y, y), axis=0)
        i = i + 1
    return x, y

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    #np.random.seed(42)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def load_tr_data(path, num_batches, lm_factor=1):
    # Train data
    x_tr_path = os.path.join(path, 'x_tr')
    x_tr = np.loadtxt(x_tr_path)
    print("x_tr shape: ", x_tr.shape)

    y_tr_path = os.path.join(path, 'y_tr')
    y_tr = np.loadtxt(y_tr_path)
    x_tr, y_tr = shuffle_in_unison(x_tr, y_tr)

    # Select landmarks from the training batches
    n_lm = int(lm_factor * np.sqrt(x_tr.shape[0]))
    lm_idx = np.random.randint(0, x_tr.shape[0], size=n_lm)
    landmarks = x_tr[lm_idx]
    landmarks = landmarks.reshape(-1, 1)
    print("number of landmarks: ", n_lm)

    x_tr_bhs = np.array(np.split(x_tr, num_batches))
    y_tr_bhs = np.array(np.split(y_tr, num_batches))

    x_tr_bhs = x_tr_bhs.reshape(num_batches, -1, 1)
    y_tr_bhs = y_tr_bhs.reshape(num_batches, -1, 1)

    return x_tr_bhs, y_tr_bhs, landmarks

def load_ts_data(path, num_ts_batches):
    x_ts_path = os.path.join(path, 'x_ts')
    x_ts = np.loadtxt(x_ts_path)
    x_ts_bhs = x_ts.reshape(num_ts_batches, -1, 1)

    y_ts_path = os.path.join(path, 'y_ts')
    y_ts = np.loadtxt(y_ts_path)
    y_ts_bhs = y_ts.reshape(num_ts_batches, -1, 1)
    return x_ts_bhs, y_ts_bhs

def training_and_validation_split(x_tr_bhs, y_tr_bhs, num_tr_batches, num_val_batches):
    # Select some of the training data batches as validation data
    x_tr_bhs, x_val_bhs, y_tr_bhs, y_val_bhs = train_test_split(x_tr_bhs, y_tr_bhs, train_size=0.8, test_size=0.2)

    x_tr_bhs = x_tr_bhs[0:num_tr_batches]
    y_tr_bhs = y_tr_bhs[0:num_tr_batches]

    x_val_bhs = x_val_bhs[0:num_val_batches]
    y_val_bhs = y_val_bhs[0:num_val_batches]

    return x_tr_bhs, y_tr_bhs, x_val_bhs, y_val_bhs

# Load falkon solver
from StreaMRAK.StreaMRAKconfig.loadConfigurations import StreaMRAKconfig
from StreaMRAK.falkonSolver import FALKON_Solver

verbosity = 1
streaMRAKconfig = StreaMRAKconfig(verbosity)
falkonSolver = FALKON_Solver(streaMRAKconfig.falkon_conf)

if __name__ == '__main__':
    ###############################
    size = -1
    DataFolder = 'VarSinus'
    ExperimentName = 'VarSinus_FALKON_choose_bw'

    total_num_tr_batches = 200
    num_val_batches = 2
    num_tr_batches = 5
    num_ts_batches = 1
    target_dim = 1
    n_bw_points = 2
    ###############################

    ### Load data
    path = os.path.join(os.getcwd(), 'Datasets', DataFolder)
    x_tr_bhs, y_tr_bhs, landmarks = load_tr_data(path, total_num_tr_batches,
                                                 streaMRAKconfig.monitor_conf['lmNodeRatioFactor'])
    x_ts_bhs, y_ts_bhs = load_ts_data(path, num_ts_batches)

    x_tr_bhs, y_tr_bhs, x_val_bhs, y_val_bhs = training_and_validation_split(x_tr_bhs, y_tr_bhs, num_tr_batches, num_val_batches)

    n_lm, d = landmarks.shape
    bw_grid = np.logspace(start=1, stop=-5, num=n_bw_points)
    mse_val_list = []
    tic = perf_counter()
    for bw in bw_grid:
        print("Bw: ", bw)
        n_tot = 0
        Kmm = np.zeros((n_lm, n_lm))
        KnmTKnm = np.zeros((n_lm, n_lm))
        Zm = np.zeros((n_lm, target_dim))

        for idx in range(0, num_tr_batches):
            x_tr_bh = x_tr_bhs[idx]
            y_tr_bh = y_tr_bhs[idx]
            print("Training Batch nr: ", idx)

            n_tot = n_tot + y_tr_bh.shape[0]

            KnmTKnm_tmp = falkonSolver.calcKnmTKnm(x_tr_bh, landmarks, bw)
            KnmTKnm = KnmTKnm + KnmTKnm_tmp

            Zm_tmp = falkonSolver.calcZm(x_tr_bh, y_tr_bh, landmarks, bw)
            Zm = Zm + Zm_tmp
        Zm = (1 / n_tot) * Zm
        Kmm = falkonSolver.calcKmm(landmarks, bw)
        alpha = falkonSolver.fit_at_scale(Kmm, KnmTKnm, Zm, n_tr=n_tot)

        mse_val = 0
        for idx_val in range(0, num_val_batches):
            print(f"Validation batch nr: {idx_val}")
            x_val = x_val_bhs[idx_val]
            y_val = y_val_bhs[idx_val]
            y_val_pred = falkonSolver.predict_at_scale(x_val, landmarks, alpha, bw)

            mse_val = mse_val + mean_squared_error(y_val, y_val_pred)
        mse_val = mse_val/num_val_batches
        mse_val_list.append(mse_val)
    opt_bw = bw_grid[mse_val_list.index(min(mse_val_list))]
    toc = perf_counter()
    time_elapse = toc-tic

    print("Bandwidth grid: ", bw_grid)
    print("MSE list ", mse_val_list)
    print("Optimal bandwidth: ", opt_bw)

    directory = os.path.join(os.getcwd(), 'LoggModel', ExperimentName)
    Path(directory).mkdir(parents=True, exist_ok=True)

    fileName_bw = 'bw_grid'
    path_bw = os.path.join(directory, fileName_bw)
    np.savetxt(path_bw, bw_grid)

    fileName_mse = 'mse_bw_grid'
    path_mse = os.path.join(directory, fileName_mse)
    np.savetxt(path_mse, mse_val_list)

    fileName_timeElapse = 'bw_time_elapse'
    path_telapse = os.path.join(directory, fileName_timeElapse)
    np.savetxt(path_telapse, np.array([time_elapse]))

    fileName_optBw = 'opt_bw'
    path_telapse = os.path.join(directory, fileName_optBw)
    np.savetxt(path_telapse, np.array([opt_bw]))
