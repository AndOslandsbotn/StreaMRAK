import numpy as np
import os
from sklearn.metrics import mean_squared_error
from time import perf_counter
from pathlib import Path

# Load falkon solver
from StreaMRAK.StreaMRAKconfig.loadConfigurations import StreaMRAKconfig
from StreaMRAK.falkonSolver import FALKON_Solver


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

def load_tr_data(path, targetType, num_batches, loopdata = False, n_loops=1):
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
    lm_factor=10
    n_lm = int(lm_factor * np.sqrt(x_tr.shape[0]))
    lm_idx = np.random.randint(0, x_tr.shape[0], size=n_lm)
    landmarks = x_tr[lm_idx]
    print("number of landmarks: ", n_lm)

    x_tr_bhs = np.array_split(x_tr, num_batches)
    y_tr_bhs = np.array_split(y_tr, num_batches)
    #x_tr_bhs = x_tr_bhs[:3]
    #y_tr_bhs = y_tr_bhs[:3] # NB! num_tr_batches in the foor loop must be set to 3...
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


if __name__ == '__main__':
    ###############################
    num_tr_batches = 200
    num_ts_batches = 1
    size = -1
    DataFolder = 'Dumbell'
    ExperimentName = 'Dumbell_FALKON'
    targetType = 'complexSinus'

    n_ts_p = 60000
    target_dim = 1
    ###############################

    verbosity = 1
    streaMRAKconfig = StreaMRAKconfig(verbosity)
    falkonSolver = FALKON_Solver(streaMRAKconfig.falkon_conf)


    ### Load data
    path = os.path.join(os.getcwd(), 'Datasets', DataFolder)
    x_tr_bhs, y_tr_bhs, landmarks = load_tr_data(path, targetType, num_tr_batches)
    x_ts_bhs, y_ts_bhs = load_ts_data(path, targetType, n_ts_p, num_ts_batches)
    x_ts = x_ts_bhs[0]
    y_ts = y_ts_bhs[0]
    n_lm, d = landmarks.shape

    # Init matrices at landmarks
    target_dim = 1
    Kmm = np.zeros((n_lm, n_lm))
    KnmTKnm = np.zeros((n_lm, n_lm))
    Zm = np.zeros((n_lm, target_dim))

    bw_opt = 1.13
    forecastStep = 10
    n_tot = 0
    start_trainTime = perf_counter()

    for idx in range(0, num_tr_batches):
        x_tr_bh = x_tr_bhs[idx]
        y_tr_bh = y_tr_bhs[idx]
        print("Batch nr: ", idx)

        n_tot = n_tot + y_tr_bh.shape[0]

        KnmTKnm_tmp = falkonSolver.calcKnmTKnm(x_tr_bh, landmarks, bw_opt)
        KnmTKnm = KnmTKnm + KnmTKnm_tmp

        Zm_tmp = falkonSolver.calcZm(x_tr_bh, y_tr_bh, landmarks, bw_opt)
        Zm = Zm + Zm_tmp
    Zm = (1/n_tot)*Zm
    Kmm = falkonSolver.calcKmm(landmarks, bw_opt)
    alpha = falkonSolver.fit_at_scale(Kmm, KnmTKnm, Zm, n_tr=n_tot)
    stop_trainTime = perf_counter()
    timeElapse_trainTime = abs(stop_trainTime-start_trainTime)

    start = perf_counter()
    y_ts_pred = falkonSolver.predict_at_scale(x_ts, landmarks, alpha, bw_opt)
    stop = perf_counter()
    timeElapse_predTime = abs(stop-start)

    mse_ts = mean_squared_error(y_ts, y_ts_pred)
    print("MSE: ", mse_ts)

    summary_dict = {}
    summary_dict['NumBatches'] = [num_tr_batches]
    summary_dict['Bandwidth'] = [bw_opt]
    summary_dict['TimeElapse_trainTime'] = [timeElapse_trainTime]
    summary_dict['TimeElapse_predTime'] = [timeElapse_predTime]

    # Save data to file
    directory = os.path.join(os.getcwd(), 'Results', ExperimentName)
    Path(directory).mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.DataFrame(data=summary_dict)
    df.to_csv(os.path.join(directory, 'summary'), sep=',', index=False)

    fileName = os.path.join(directory, 'x_ts.npy')
    np.save(fileName, x_ts_bhs)

    fileName = os.path.join(directory, 'y_ts.npy')
    np.save(fileName, y_ts_bhs)

    fileName = os.path.join(directory, 'y_ts_pred.npy')
    np.save(fileName, y_ts_pred)

    fileName = os.path.join(directory, 'landmarks.npy')
    np.save(fileName, landmarks)

