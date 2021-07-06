import numpy as np
import os
from sklearn.metrics import mean_squared_error
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


from StreaMRAK.StreaMRAKmain import DataLoader2
from MRFALKON.MRFALKONmainFULLDATASET import MRFALKON_FULLDATASET_Master, MRFALKONconfig

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

def load_tr_data(path, targetType, num_batches, loopdata = False, n_loops=1, lm_factor=1):
    # Train data
    x_tr_path = os.path.join(path, 'twoblobTr.csv')
    x_tr = np.genfromtxt(x_tr_path, delimiter=',')
    print("x shape: ", x_tr.shape)

    y_tr_path = os.path.join(path, targetType+'Tr.csv')
    y_tr = np.genfromtxt(y_tr_path, delimiter=',')
    y_tr = y_tr.reshape(-1, 1)
    print("y shape: ", y_tr.shape)

    if loopdata == True:
        x_tr, y_tr = loop_data(x_tr, y_tr, n_loops)

    x_tr, y_tr = shuffle_in_unison(x_tr, y_tr)

    # Select landmarks from the training batches
    n_lm = int(lm_factor * np.sqrt(x_tr.shape[0]))
    lm_idx = np.random.randint(0, x_tr.shape[0], size=n_lm)
    landmarks = x_tr[lm_idx]
    print("number of landmarks: ", n_lm)

    x_tr_bhs = np.array_split(x_tr, num_batches)
    y_tr_bhs = np.array_split(y_tr, num_batches)
    #x_tr_bhs = x_tr_bhs[:5]
    #y_tr_bhs = y_tr_bhs[:5]
    #landmarks = landmarks[:200]
    return x_tr_bhs, y_tr_bhs, landmarks

def load_ts_data(path, targetType, n_ts_p, num_ts_batches):
    # Test data
    x_ts_path = os.path.join(path, 'twoblobTs.csv')
    x_ts = np.genfromtxt(x_ts_path, delimiter=',')
    x_ts = x_ts[0:n_ts_p, :]
    x_ts_bhs = np.array_split(x_ts, num_ts_batches)

    y_ts_path = os.path.join(path, targetType+'Ts.csv')
    y_ts = np.genfromtxt(y_ts_path, delimiter=',')
    y_ts = y_ts.reshape(-1, 1)
    y_ts = y_ts[0:n_ts_p, :]
    y_ts_bhs = np.array_split(y_ts, num_ts_batches)
    return x_ts_bhs, y_ts_bhs


from time import perf_counter
if __name__ == '__main__':

    #####################################
    num_tr_batches = 20
    num_ts_batches = 10
    n_ts_p = 100000
    DataFolder = 'Dumbell'
    ExperimentName = 'Dumbell_MRFALKON_fulldataset'
    targetType = 'complexSinus'
    maxLvl = 12
    ######################################

    # Configurations
    verbosity = 1
    mrfalkonconfig = MRFALKONconfig(verbosity, tag='_mrfalkon')

    ### Load data
    path = os.path.join(os.getcwd(), 'Datasets', DataFolder)
    x_tr_bhs, y_tr_bhs, landmarks = load_tr_data(path, targetType, num_tr_batches, mrfalkonconfig.MRFALKONmainConfig['lmNodeRatioFactor'])
    x_ts_bhs, y_ts_bhs = load_ts_data(path, targetType, n_ts_p, num_ts_batches)
    dataLoader = DataLoader2(X_batches=x_tr_bhs, y_batches=y_tr_bhs)

    ### Setup the StreaMRAK algorithm
    initPoint = x_tr_bhs[0][0, :]
    initTarget = y_tr_bhs[0][0, :]
    initRadius = 10
    initPath = ()
    mrfalkon = MRFALKON_FULLDATASET_Master(initRadius, landmarks, mrfalkonconfig, ExperimentName)
    #########################################################################################################

    start = perf_counter()
    mrfalkon.learnFromStream(dataLoader=dataLoader, ExpName=ExperimentName, logg=True, maxLvl=maxLvl)
    stop = perf_counter()
    print("Total time elapse: ", stop - start)

    list_of_fitted_lvls = mrfalkon.multiResModel.list_of_fitted_levels
    max_lvl = list_of_fitted_lvls[-1]

    mse_ts_ll = []
    print("Prediction phase")
    for bhs in range(0, num_ts_batches):
        y_ts_pred_list = []
        mse_ts_list = []
        for lvl in range(0, max_lvl+1):
            y_ts_pred = mrfalkon.predict(x_ts_bhs[bhs], max_lvl=lvl)
            y_ts_pred_list.append(y_ts_pred.flatten())

            mse_ts = mean_squared_error(y_ts_bhs[bhs], y_ts_pred)
            print(f'MSE at lvl {lvl}: {mse_ts}')
            mse_ts_list.append(mse_ts)
        # Save predictions to file

        directory = os.path.join(os.getcwd(), 'Results', ExperimentName)
        Path(directory).mkdir(parents=True, exist_ok=True)

        path_y_pred = os.path.join(directory, f'y_ts_pred_bhs{bhs}')
        y_ts_pred_arr = np.array(y_ts_pred_list).transpose()
        np.savetxt(path_y_pred, y_ts_pred_arr, delimiter=',')

        path_x_ts_bhs = os.path.join(directory, f'x_ts_bhs{bhs}')
        np.savetxt(path_x_ts_bhs, x_ts_bhs[bhs], delimiter=',')

        path_y_ts_bhs = os.path.join(directory, f'y_ts_bhs{bhs}')
        np.savetxt(path_y_ts_bhs, y_ts_bhs[bhs].flatten(), delimiter=',')

        mse_ts_ll.append(mse_ts_list)
    mse_ts_arr = np.array(mse_ts_ll)
    avg_mse_ts_list = np.mean(mse_ts_arr, axis=0)
    mrfalkon.store_model_summary(directory, mse_ts_list, ExperimentName)
    mrfalkonconfig.store_config(directory)

    directory = os.path.join(os.getcwd(), 'LoggModel', ExperimentName)
    Path(directory).mkdir(parents=True, exist_ok=True)
    path_lm = os.path.join(directory, 'mrfalkon_landmarks')
    np.savetxt(path_lm, landmarks, delimiter=',')
    print("DONE")