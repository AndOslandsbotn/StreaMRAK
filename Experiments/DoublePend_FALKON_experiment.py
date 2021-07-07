import os
import numpy as np
from sklearn.metrics import mean_squared_error
from time import perf_counter
from pathlib import Path

from Experiments.Datasets.DPmain import calc_DP_poss, calc_DP_cm
from StreaMRAK.StreaMRAKconfig.loadConfigurations import loadConfigFromCSV

def load_ts_data(path):
    # Load test data
    fileName = os.path.join(path, 'state_ts.npy')
    state_ts = np.load(fileName)

    fileName = os.path.join(path, 'cm_ts.npy')
    cm_ts = np.load(fileName)

    fileName = os.path.join(path, 'poss_ts.npy')
    poss_ts = np.load(fileName)
    return state_ts, cm_ts, poss_ts

def load_tr_data(path, recording_length, num_batches, num_debug_batches, forecastStep):
    # Load training data
    fileName = os.path.join(path, 'state_tr.npy')
    state_tr = np.load(fileName)

    state_tr = state_tr[:, :recording_length, :]
    state_tr_bhs = np.array(np.split(state_tr, num_batches))


    state_tr_bhs = state_tr_bhs[0:num_debug_batches]

    state_tr_bhs = state_tr_bhs[:, :, :-forecastStep, :]

    # Select landmarks from the training batches
    potential_lm = state_tr_bhs.reshape(-1, state_tr_bhs.shape[-1])
    n_lm = 10 * int(np.sqrt(potential_lm.shape[0]))
    lm_idx = np.random.randint(0, potential_lm.shape[0], size=n_lm)
    landmarks = potential_lm[lm_idx, :]

    #state_tr_bhs = state_tr_bhs[:5]
    return state_tr_bhs, landmarks


# Load falkon solver
from StreaMRAK.StreaMRAKconfig.loadConfigurations import StreaMRAKconfig
from StreaMRAK.falkonSolver import FALKON_Solver
verbosity = 1
streaMRAKconfig = StreaMRAKconfig(verbosity)
falkonSolver = FALKON_Solver(streaMRAKconfig.falkon_conf)

if __name__ == '__main__':
    ####################################################################
    pendulum_energies = ['DoublePendLowEnergy', 'DoublePendHighEnergy']
    pendulum_energy = pendulum_energies[0]
    #ExperimentName = 'Dp_HighE_FALKON'
    ExperimentName = 'Dp_LowE_FALKON_test'
    config_tag = '_doublePend'
    recording_length = 500
    num_batches = 200
    num_debug_batches = 10
    num_test_pendulums = 1
    forecastStep = 10
    verbosity = 1
    #####################################################################

    # Load data
    path = os.path.join('Datasets', pendulum_energy)
    state_ts, cm_ts, poss_ts = load_ts_data(path)
    state_tr_bhs, landmarks = load_tr_data(path, recording_length, num_batches, num_debug_batches, forecastStep)

    # Init matrices at landmarks
    n_lm, d = landmarks.shape

    # Save data to file
    directory = os.path.join(os.getcwd(), 'Results', ExperimentName)
    Path(directory).mkdir(parents=True, exist_ok=True)
    np.savetxt(os.path.join(os.getcwd(), 'Results', ExperimentName, 'num_lm'), np.array([n_lm]))
    print("n_lm: ", n_lm)

    target_dim = 4
    Kmm = np.zeros((n_lm, n_lm))
    KnmTKnm = np.zeros((n_lm, n_lm))
    Zm = np.zeros((n_lm, target_dim))

    if ExperimentName == 'Dp_LowE_FALKON':
        bw_opt = 0.784
    elif ExperimentName == 'Dp_HighE_FALKON':
        bw_opt = 23.35
    forecastStep = 10
    bhnr = 0
    n_tot = 0
    start_trainTime = perf_counter()
    for state_tr in state_tr_bhs:
        print("Batch nr: ", bhnr)
        bhnr += 1
        y_tr = state_tr[:, forecastStep:, :]
        state_tr = state_tr[:, :-forecastStep, :]

        state_tr = state_tr.reshape(-1, state_tr.shape[-1])
        y_tr = y_tr.reshape(-1, y_tr.shape[-1])
        n_tot = n_tot + y_tr.shape[0]
        print("n_tot:", n_tot)

        KnmTKnm_tmp = falkonSolver.calcKnmTKnm(state_tr, landmarks, bw_opt)
        KnmTKnm = KnmTKnm + KnmTKnm_tmp

        Zm_tmp = falkonSolver.calcZm(state_tr, y_tr, landmarks, bw_opt)
        Zm = Zm + Zm_tmp
    Zm = (1/n_tot)*Zm
    Kmm = falkonSolver.calcKmm(landmarks, bw_opt)
    alpha = falkonSolver.fit_at_scale(Kmm, KnmTKnm, Zm, n_tr=n_tot)
    stop_trainTime = perf_counter()
    timeElapse_trainTime = stop_trainTime-start_trainTime

    ## Predictions
    dpModelConfig = loadConfigFromCSV('main', config_tag, verbosity)
    L1 = dpModelConfig['L1']  # length of pendulum 1 in m
    L2 = dpModelConfig['L2']  # length of pendulum 2 in m
    M1 = dpModelConfig['M1']  # mass of pendulum 1 in kg
    M2 = dpModelConfig['M2']  # mass of pendulum 2 in kg

    nbhs, ntsp, d = state_ts.shape
    nsteps = int(ntsp/forecastStep)

    start_predTime = perf_counter()
    fstate_ts_pred_ll = []
    fposs_ll = []
    cposs_ll = []
    fcm_ll = []
    ccm_ll = []
    mse_ts_ll = []
    for test_pend_nr in range(0, num_test_pendulums):
        fstate_ts_pred_list = []
        fposs_list = []
        cposs_list = []
        fcm_list = []
        ccm_list = []
        mse_ts_list = []

        fstate_ts_pred = np.array([state_ts[test_pend_nr, 0, :]])
        for step in range(1, nsteps):
            fstate_ts_pred = falkonSolver.predict_at_scale(fstate_ts_pred, landmarks, alpha, bw_opt)
            fstate_ts_pred_list.append(fstate_ts_pred[0])

            fposs = calc_DP_poss(L1, L2, fstate_ts_pred[0])
            fposs_list.append(fposs)

            fcm = calc_DP_cm(M1, M2, fposs)
            fcm_list.append(fcm)

            # Calc cm and mass positions from correct state
            cposs = calc_DP_poss(L1, L2, state_ts[test_pend_nr, step*forecastStep, :])
            cposs_list.append(cposs)

            ccm = calc_DP_cm(M1, M2, cposs)
            ccm_list.append(ccm)

            mse_ts = mean_squared_error(state_ts[test_pend_nr, step * forecastStep, :], fstate_ts_pred[0])
            mse_ts_list.append(mse_ts)
            print(mse_ts)

        fstate_ts_pred_ll.append(fstate_ts_pred_list)
        fposs_ll.append(fposs_list)
        cposs_ll.append(cposs_list)
        fcm_ll.append(fcm_list)
        ccm_ll.append(ccm_list)
        mse_ts_ll.append(mse_ts_list)
    fstate_ts_pred_arr = np.array(fstate_ts_pred_ll)
    fposs_arr = np.array(fposs_ll)
    cposs_arr = np.array(cposs_ll)
    fcm_arr = np.array(fcm_ll)
    ccm_arr = np.array(ccm_ll)
    mse_ts_arr = np.array(mse_ts_ll)
    avg_mse_ts = np.mean(mse_ts_arr, axis=0)

    stop_predTime = perf_counter()
    timeElapse_predTime = stop_predTime-start_predTime
    import pandas as pd

    summary_dict = {}
    summary_dict['NumBatches'] = [num_batches]
    summary_dict['ForecastStepSize'] = [forecastStep]
    summary_dict['NumForecastSteps'] = [nsteps]
    summary_dict['Bandwidth'] = [bw_opt]
    summary_dict['TimeElapse_trainTime'] = [timeElapse_trainTime]
    summary_dict['TimeElapse_predTime'] = [timeElapse_predTime]

    # Save data to file
    directory = os.path.join(os.getcwd(), 'Results', ExperimentName)
    Path(directory).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(data=summary_dict)
    df.to_csv(os.path.join(directory, 'summary'), sep=',', index=False)

    fileName = os.path.join(directory, 'fstate_ts_pred.npy')
    np.save(fileName, fstate_ts_pred_arr)

    fileName = os.path.join(directory, 'fcm_ts_pred.npy')
    np.save(fileName, fcm_arr)

    fileName = os.path.join(directory, 'ccm_ts_pred.npy')
    np.save(fileName, ccm_arr)

    fileName = os.path.join(directory, 'fposs_ts_pred.npy')
    np.save(fileName, fposs_arr)

    fileName = os.path.join(directory, 'cposs_ts_pred.npy')
    np.save(fileName, cposs_arr)

    fileName = os.path.join(directory, 'landmarks.npy')
    np.save(fileName, landmarks)


