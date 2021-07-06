import numpy as np
import os
from sklearn.metrics import mean_squared_error
from pathlib import Path

from StreaMRAK.StreaMRAKmain import StreaMRAKconfig, StreaMRAKmaster
from StreaMRAK.StreaMRAKmain import DataLoader

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

def load_tr_data(path, num_batches):
    # Train data
    x_tr_path = os.path.join(path, 'x_tr')
    x_tr = np.loadtxt(x_tr_path)
    print("x_tr shape: ", x_tr.shape)

    y_tr_path = os.path.join(path, 'y_tr')
    y_tr = np.loadtxt(y_tr_path)
    x_tr, y_tr = shuffle_in_unison(x_tr, y_tr)

    x_tr_bhs = np.array(np.split(x_tr, num_batches))
    y_tr_bhs = np.array(np.split(y_tr, num_batches))

    x_tr_bhs = x_tr_bhs.reshape(num_batches, -1, 1)
    y_tr_bhs = y_tr_bhs.reshape(num_batches, -1, 1)

    #x_tr_bhs = x_tr_bhs[:5, :, :]
    #y_tr_bhs = y_tr_bhs[:5, :, :]
    return x_tr_bhs, y_tr_bhs

def load_ts_data(path, num_ts_batches):
    x_ts_path = os.path.join(path, 'x_ts')
    x_ts = np.loadtxt(x_ts_path)
    x_ts_bhs = x_ts.reshape(num_ts_batches, -1, 1)

    y_ts_path = os.path.join(path, 'y_ts')
    y_ts = np.loadtxt(y_ts_path)
    y_ts_bhs = y_ts.reshape(num_ts_batches, -1, 1)
    return x_ts_bhs, y_ts_bhs

from time import perf_counter
if __name__ == '__main__':
    ###############################
    num_batches = 200
    num_ts_batches = 10
    size = -1
    DataFolder = 'VarSinus'
    ExperimentName = 'VarSinus_StreaMRAK'
    ###############################

    ### Load data
    path = os.path.join(os.getcwd(), '..', 'Datasets', DataFolder)
    x_tr_bhs, y_tr_bhs = load_tr_data(path, num_batches)
    x_ts_bhs, y_ts_bhs = load_ts_data(path, num_ts_batches)
    dataLoader = DataLoader(X_batches=x_tr_bhs, y_batches=y_tr_bhs)

    ### Setup the StreaMRAK algorithm
    verbosity = 1
    streaMRAKconfig = StreaMRAKconfig(verbosity, tag='_varsin')
    initPoint = x_tr_bhs[0][0]
    initTarget = y_tr_bhs[0][0]
    initRadius = 1
    initPath = ()
    streaMRAK = StreaMRAKmaster(initPoint, initTarget, initRadius, initPath, streaMRAKconfig, ExperimentName)
    #########################################################################################################

    start = perf_counter()
    streaMRAK.learnFromStream(dataLoader=dataLoader, ExpName=ExperimentName, logg=True)

    stop = perf_counter()
    print("Total time elapse: ", stop - start)
    list_of_fitted_lvls = streaMRAK.multiResModel.list_of_fitted_levels
    max_lvl = list_of_fitted_lvls[-1]

    mse_ts_ll = []
    for bhs in range(0, num_ts_batches):
        y_ts_pred_list = []
        mse_ts_list = []
        for lvl in range(0, max_lvl+1):

            y_ts_pred = streaMRAK.predict(x_ts_bhs[bhs, :, :], max_lvl=lvl)
            y_ts_pred_list.append(y_ts_pred.flatten())

            mse_ts = mean_squared_error(y_ts_bhs[bhs, :, :], y_ts_pred)
            mse_ts_list.append(mse_ts)
            print(f'MSE at lvl {lvl}: {mse_ts}')

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
    streaMRAK.store_model_summary(directory, avg_mse_ts_list, ExperimentName)
    streaMRAKconfig.store_config(directory)
