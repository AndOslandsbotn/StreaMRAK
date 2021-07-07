import numpy as np
import os
from pathlib import Path

from StreaMRAK.StreaMRAKconfig.loadConfigurations import loadConfigFromCSV
from Experiments.Datasets.DPmain import DoublePendulumModel

##################
###### MENU ######
##################
num_ts_points = 2000
num_init_cond = 8000
stepMax = 500
eps = 10**(-2)
low_E = False

if low_E:
    path = os.path.join(os.getcwd(), 'DoublePendLowEnergy')
    Path(path).mkdir(parents=True, exist_ok=True)
    # Init cond
    th1 = -20
    th2 = -20
    w1 = 0
    w2 = 0
    maxAng = 3
    maxAngVelo = 3
else:
    path = os.path.join(os.getcwd(), 'DoublePendHighEnergy')
    Path(path).mkdir(parents=True, exist_ok=True)
    # Init cond
    th1 = -120
    th2 = -20
    w1 = -7.57
    w2 = 7.68
    maxAng = 3
    maxAngVelo = 3
verbosity = 1

#########################################################################
####################### Generate training dataset   #####################
#########################################################################
fileName = 'DPconfig'
fileNameTag = '_doublePend'

dpModelConfig = loadConfigFromCSV('main', '_doublePend', verbosity)
dp = DoublePendulumModel(dpModelConfig)

initialConditions = np.zeros(4)
current_state_listOflist = []
current_cm_listOflist = []
curr_poss_listOflist = []
for t in range(0, num_init_cond):
    print(f"\nInitial condition nr: {t}")

    #initialConditions[0] = th1 + np.random.uniform(-0.025*th1, 0.025*th1)
    #initialConditions[1] = th2 + np.random.uniform(-0.15*th2, 0.15*th2)
    #initialConditions[2] = w1 + np.random.uniform(-0.3*w1, 0.3*w1)
    #initialConditions[3] = w2 + np.random.uniform(-0.3*w2, 0.3*w2)

    initialConditions[0] = th1 + np.random.normal(0, abs(0.025*th1))
    initialConditions[1] = th2 + np.random.normal(0, abs(0.15*th2))
    initialConditions[2] = w1 + np.random.normal(0, abs(0.3*w1))
    initialConditions[3] = w2 + np.random.normal(0, abs(0.3*w2))
    dp.reset_state(initialConditions)

    current_state_list = []
    current_cm_list = []
    curr_poss_list = []
    for step in range(0, stepMax+1):
        if step % 100 == 0:
            print(f"Step nr: {step}")
        dp.makeStep()
        current_state = dp.get_current_state()
        current_cm = dp.get_current_CM_poss()
        curr_poss = dp.get_current_mass_poss()

        current_state_list.append(current_state[0])
        current_cm_list.append(current_cm)
        curr_poss_list.append(curr_poss)
    current_state_listOflist.append(current_state_list)
    current_cm_listOflist.append(current_cm_list)
    curr_poss_listOflist.append(curr_poss_list)
current_state = np.array(current_state_listOflist)
current_cm = np.array(current_cm_listOflist)
curr_poss = np.array(curr_poss_listOflist)


fileName = os.path.join(path, 'state_tr')
np.save(fileName, current_state)

fileName = os.path.join(path, 'cm_tr')
np.save(fileName, current_cm)

fileName = os.path.join(path, 'poss_tr')
np.save(fileName, curr_poss)

#####################################################################
####################### Generate test dataset   #####################
#####################################################################
num_ts_pendulums = 100
current_state_ll = []
current_cm_ll = []
curr_poss_ll = []
for ts_pend in range(0, num_ts_pendulums):
    dp_test = DoublePendulumModel(dpModelConfig)
    initialConditions_ts = np.zeros(4)
    #initialConditions_ts[0] = th1 + np.random.uniform(-eps*th1, eps*th1)
    #initialConditions_ts[1] = th2 + np.random.uniform(-eps*th2, eps*th2)
    #initialConditions_ts[2] = w1 + np.random.uniform(-eps*w1, eps*w1)
    #initialConditions_ts[3] = w2 + np.random.uniform(-eps*w2, eps*w2)

    initialConditions_ts[0] = th1 + np.random.normal(0, abs(eps*th1))
    initialConditions_ts[1] = th2 + np.random.normal(0, abs(eps*th2))
    initialConditions_ts[2] = w1 + np.random.normal(0, abs(eps*w1))
    initialConditions_ts[3] = w2 + np.random.normal(0, abs(eps*w2))
    dp_test.reset_state(initialConditions_ts)

    current_state_list = []
    current_cm_list = []
    curr_poss_list = []
    for step in range(0, stepMax):
        dp_test.makeStep()
        current_state = dp_test.get_current_state()
        current_cm = dp_test.get_current_CM_poss()
        curr_poss = dp_test.get_current_mass_poss()

        # Noise for all pendulums except the first one
        if ts_pend > 0:
            current_state = current_state + (current_state * np.random.uniform(-eps, eps))
            current_cm = current_cm + (current_cm * np.random.uniform(-eps, eps))
            curr_poss = curr_poss + (curr_poss * np.random.uniform(-eps, eps))

        current_state_list.append(current_state[0])
        current_cm_list.append(current_cm)
        curr_poss_list.append(curr_poss)
    current_state_ll.append(current_state_list)
    current_cm_ll.append(current_cm_list)
    curr_poss_ll.append(curr_poss_list)

current_state = np.array(current_state_ll)
current_cm = np.array(current_cm_ll)
curr_poss = np.array(curr_poss_ll)

fileName = os.path.join(path, 'state_ts')
np.save(fileName, current_state)

fileName = os.path.join(path, 'cm_ts')
np.save(fileName, current_cm)

fileName = os.path.join(path, 'poss_ts')
np.save(fileName, curr_poss)

# Example of reading
fileName = os.path.join(path, 'state_tr.npy')
curr_state_read = np.load(fileName)
print(curr_state_read.shape)