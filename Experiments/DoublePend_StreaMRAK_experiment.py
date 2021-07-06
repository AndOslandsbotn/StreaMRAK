import os
import numpy as np
from sklearn.metrics import mean_squared_error
from time import perf_counter
from pathlib import Path

from StreaMRAK.StreaMRAKmain import DataLoader

from DoublePendulum.doublePendulumAuxillary import calcEnergy, calcConstEnergyDomainBoundaries
from DoublePendulum.DPconfig import loadConfig
from DoublePendulum.DPmain import calc_DP_poss, calc_DP_cm

from StreaMRAK.StreaMRAKmain import StreaMRAKconfig, StreaMRAKmaster
from StreaMRAK.StreaMRAKconfig.loadConfigurations import loadConfigFromCSV

def double_pendulum_config(state_ts):
    def distance(th1, th2, w1, w2):
        return np.sqrt((2 * th1) ** 2 + (2 * th2) ** 2 + (2 * w1) ** 2 + (2 * w2) ** 2)

    fileNameTag = '_doublePend'
    fileName = 'main'
    verbosity = 1
    dpModelConfig = loadConfigFromCSV(fileName, fileNameTag, verbosity)

    E_I = calcEnergy(state_ts[0], dpModelConfig)
    th1_max, th2_max, w1_max, w2_max = calcConstEnergyDomainBoundaries(E_I, dpModelConfig)

    maxDistance = distance(th1_max, th2_max, w1_max, w2_max)
    return dpModelConfig, maxDistance

def load_ts_data(path):
    # Load test data
    fileName = os.path.join(path, 'state_ts.npy')
    state_ts = np.load(fileName)

    fileName = os.path.join(path, 'cm_ts.npy')
    cm_ts = np.load(fileName)

    fileName = os.path.join(path, 'poss_ts.npy')
    poss_ts = np.load(fileName)
    return state_ts, cm_ts, poss_ts

def load_tr_data(path, recording_length, num_batches, forecastStep):
    # Load training data
    fileName = os.path.join(path, 'state_tr.npy')
    state_tr = np.load(fileName)

    state_tr = state_tr[:, :recording_length, :]
    state_tr_bhs = np.array(np.split(state_tr, num_batches))

    y_tr_bhs = state_tr_bhs[:, :, forecastStep:, :]
    state_tr_bhs = state_tr_bhs[:, :, :-forecastStep, :]

    state_tr_bhs = state_tr_bhs.reshape(num_batches, -1, state_tr.shape[-1])
    y_tr_bhs = y_tr_bhs.reshape(num_batches, -1, y_tr_bhs.shape[-1])

    #state_tr_bhs = state_tr_bhs[:2]
    #y_tr_bhs = y_tr_bhs[:2]
    return state_tr_bhs, y_tr_bhs

def run_double_pendulum_prediction(max_lvl, recording_length, state_ts_bhs, forecastStep, dpModelConfig, directory):
    num_test_pendulums = len(state_ts_bhs)
    max_step = int(recording_length/forecastStep)

    L1 = dpModelConfig['L1']  # length of pendulum 1 in m
    L2 = dpModelConfig['L2']  # length of pendulum 2 in m
    M1 = dpModelConfig['M1']  # mass of pendulum 1 in kg
    M2 = dpModelConfig['M2']  # mass of pendulum 2 in kg


    mse_ts_lll = []
    for test_pend_nr in range(0, num_test_pendulums):
        state_ts = state_ts_bhs[test_pend_nr]
        fstate_ts_pred_ll = []
        fposs_ll = []
        cposs_ll = []
        fcm_ll = []
        ccm_ll = []
        mse_ts_ll = []
        for lvl in range(0, max_lvl+1):
            print(" ")
            print(f"New lvl is {lvl}")
            fstate_ts_pred_list = []
            fposs_list = []
            cposs_list = []
            fcm_list = []
            ccm_list = []
            mse_ts_list = []
            fstate_ts_pred = np.array([state_ts[0, :]])
            for step in range(1, max_step):
                fstate_ts_pred = streaMRAK.predict(fstate_ts_pred, max_lvl=lvl)
                fstate_ts_pred_list.append(fstate_ts_pred[0])

                fposs = calc_DP_poss(L1, L2, fstate_ts_pred[0])
                fposs_list.append(fposs)

                fcm = calc_DP_cm(M1, M2, fposs)
                fcm_list.append(fcm)

                # Calc cm and mass positions from correct state
                cposs = calc_DP_poss(L1, L2, state_ts[step * forecastStep, :])
                cposs_list.append(cposs)

                ccm = calc_DP_cm(M1, M2, cposs)
                ccm_list.append(ccm)

                mse_ts = mean_squared_error(state_ts[step * forecastStep, :], fstate_ts_pred[0])
                mse_ts_list.append(mse_ts)
                print(f"MSE ts, at lvl {lvl}, step {step}: {mse_ts}")
            mse_ts_ll.append(mse_ts_list)
            fstate_ts_pred_ll.append(fstate_ts_pred_list)
            fposs_ll.append(fposs_list)
            cposs_ll.append(cposs_list)
            fcm_ll.append(fcm_list)
            ccm_ll.append(ccm_list)
        mse_ts_lll.append(mse_ts_ll)
        fstate_ts_pred_arr = np.array(fstate_ts_pred_ll)
        fposs_arr = np.array(fposs_ll)
        cposs_arr = np.array(cposs_ll)
        fcm_arr = np.array(fcm_ll)
        ccm_arr = np.array(ccm_ll)

        fileName = os.path.join(directory, f'fstate_ts_pred_nr{test_pend_nr}.npy')
        np.save(fileName, fstate_ts_pred_arr)

        fileName = os.path.join(directory, f'fcm_ts_pred_nr{test_pend_nr}.npy')
        np.save(fileName, fcm_arr)

        fileName = os.path.join(directory, f'ccm_ts_pred_nr{test_pend_nr}.npy')
        np.save(fileName, ccm_arr)

        fileName = os.path.join(directory, f'fposs_ts_pred_nr{test_pend_nr}.npy')
        np.save(fileName, fposs_arr)

        fileName = os.path.join(directory, f'cposs_ts_pred_nr{test_pend_nr}.npy')
        np.save(fileName, cposs_arr)

    mse_ts_arr = np.array(mse_ts_lll)
    avg_mse_ts = np.mean(mse_ts_arr, axis=0)
    streaMRAK.store_model_summary(directory, avg_mse_ts, ExperimentName)

if __name__ == '__main__':
    ####################################################################
    pendulum_energies = ['DoublePendLowEnergy', 'DoublePendHighEnergy']
    pendulum_energy = pendulum_energies[0]
    #ExperimentName = 'Dp_HighE_StreaMRAK'
    ExperimentName = 'Dp_LowE_StreaMRAK'
    config_tag = '_doublePend'
    recording_length = 500
    num_tr_batches = 200
    num_ts_batches = 10
    forecastStep = 10
    verbosity = 1
    #####################################################################

    # Load data
    path = os.path.join(os.getcwd(), 'Datasets', pendulum_energy)
    state_tr_bhs, y_tr_bhs = load_tr_data(path, recording_length, num_tr_batches, forecastStep)
    state_ts_bhs, cm_ts_bhs, poss_ts_bhs = load_ts_data(path)
    dataLoader = DataLoader(X_batches=state_tr_bhs, y_batches=y_tr_bhs)

    ### Setup the double pendulum
    dpModelConfig, maxDistance = double_pendulum_config(state_ts_bhs[0])
    if pendulum_energy == 'DoublePendLowEnergy':
        maxDistance = 1 * maxDistance
    elif pendulum_energy == 'DoublePendHighEnergy':
        maxDistance = 3 * maxDistance
        print("maxDistance; ", maxDistance)

    ### Setup the StreaMRAK algorithm
    streaMRAKconfig = StreaMRAKconfig(verbosity, tag=config_tag)
    initPoint = np.radians(state_ts_bhs[0, 0])
    initTarget = np.radians(state_ts_bhs[0, 10])
    initRadius = maxDistance
    print("initRadius; ", initRadius)
    initPath = ()

    streaMRAK = StreaMRAKmaster(initPoint, initTarget, initRadius, initPath, streaMRAKconfig, ExperimentName)

    directory = os.path.join(os.getcwd(), 'Results', ExperimentName)
    Path(directory).mkdir(parents=True, exist_ok=True)
    streaMRAKconfig.store_config(directory)
    #########################################################################################################

    ### Train the model
    start = perf_counter()
    streaMRAK.learnFromStream(dataLoader=dataLoader, ExpName=ExperimentName, logg=True)
    stop = perf_counter()
    print("Total training time elapse: ", stop-start)

    ### Make predictions
    list_of_fitted_lvls = streaMRAK.multiResModel.list_of_fitted_levels
    max_lvl = list_of_fitted_lvls[-1]

    start_pred = perf_counter()
    run_double_pendulum_prediction(max_lvl, recording_length, state_ts_bhs, forecastStep, dpModelConfig, directory)
    end_pred = perf_counter()
    print(f'Total prediction time elapse: {end_pred-start_pred}')
