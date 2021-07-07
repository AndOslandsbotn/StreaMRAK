from numpy import sin, cos
import numpy as np
import scipy.integrate as integrate

import csv
import os
from os import path

dir_path = os.path.dirname(os.path.realpath(__file__))

import matplotlib
matplotlib.use("TkAgg")

class DoublePendulumModel():
    def __init__(self, dpModelConfig):
        self.G = dpModelConfig['G']  # acceleration due to gravity, in m/s^2
        self.L1 = dpModelConfig['L1']  # length of pendulum 1 in m
        self.L2 = dpModelConfig['L2']  # length of pendulum 2 in m
        self.M1 = dpModelConfig['M1']  # mass of pendulum 1 in kg
        self.M2 = dpModelConfig['M2']  # mass of pendulum 2 in kg
        self.dt = dpModelConfig['dt']

        self.th1 = dpModelConfig['th1']  # Initial theta1
        self.th2 = dpModelConfig['th2']  # Initial theta2
        self.w1 = dpModelConfig['w1']  # Initial omega1
        self.w2 = dpModelConfig['w2']  # Initial omega2

        self.state = np.radians([[self.th1, self.th2, self.w1, self.w2]])  # Initialize state
        self.stateHistory = [self.state[0]]

        self.curr_mass_poss = np.array([0, 0, 0, 0])

    def reset_state(self, newState):
        th1, th2, w1, w2 = newState
        self.state = np.radians([[th1, th2, w1, w2]])

    # dynamics
    def derivs(self, state, t):
        dxdt = np.zeros_like(state)
        dxdt[0] = state[2]  # set dTheta1/dt = w1
        delta = state[1] - state[0]
        den1 = (self.M1 + self.M2) * self.L1 - self.M2 * self.L1 * cos(delta) * cos(delta)
        dxdt[2] = ((self.M2 * self.L1 * state[2] * state[2] * sin(delta) * cos(delta)
                    + self.M2 * self.G * sin(state[1]) * cos(delta)
                    + self.M2 * self.L2 * state[3] * state[3] * sin(delta)
                    - (self.M1 + self.M2) * self.G * sin(state[0])) / den1)  # dw1/dt
        dxdt[1] = state[3]  # Set dTheta2/dt = w2
        den2 = (self.L2 / self.L1) * den1
        dxdt[3] = ((- self.M2 * self.L2 * state[3] * state[3] * sin(delta) * cos(delta)
                    + (self.M1 + self.M2) * self.G * sin(state[0]) * cos(delta)
                    - (self.M1 + self.M2) * self.L1 * state[2] * state[2] * sin(delta)
                    - (self.M1 + self.M2) * self.G * sin(state[1])) / den2)  # dw2/dt
        return dxdt

    def makeStep(self):
        # integrate your ODE using scipy.integrate.
        self.state = integrate.odeint(self.derivs, self.state[-1], np.array([0, self.dt]))
        self.stateHistory.append(self.state[-1])

        # get positions of the two masses
        x1 = self.L1 * sin(self.state[-1, 0])
        y1 = -self.L1 * cos(self.state[-1, 0])
        x2 = self.L2 * sin(self.state[-1, 1]) + x1
        y2 = -self.L2 * cos(self.state[-1, 1]) + y1
        self.curr_mass_poss = np.array([x1, y1, x2, y2])

    def get_current_state(self):
        return np.array([self.state[-1]])

    def get_prev_state(self, numTimeStepsBack):
        try:
            numTimeStepsBack < len(self.stateHistory)
        except:
            ValueError('No recorded history this far back in time')
        return np.array([self.stateHistory[-numTimeStepsBack]])

    def get_current_mass_poss(self):
        return self.curr_mass_poss

    def get_current_CM_poss(self):
        x_cm = (self.curr_mass_poss[0] * self.M1 + self.curr_mass_poss[2] * self.M2)/(self.M1 + self.M2)
        y_cm = (self.curr_mass_poss[1] * self.M1 + self.curr_mass_poss[3] * self.M2)/(self.M1 + self.M2)
        return np.array([x_cm, y_cm])

    def get_current_CM_x(self):
        x_cm = (self.curr_mass_poss[0] * self.M1 + self.curr_mass_poss[2] * self.M2)/(self.M1 + self.M2)
        return np.array([x_cm])

def calc_DP_poss(L1, L2, state):
    # get positions of the two masses
    x1 = L1 * sin(state[0])
    y1 = -L1 * cos(state[0])
    x2 = L2 * sin(state[1]) + x1
    y2 = -L2 * cos(state[1]) + y1
    curr_mass_poss = np.array([x1, y1, x2, y2])
    return curr_mass_poss

def calc_DP_cm(M1, M2, curr_mass_poss):
    x_cm = (curr_mass_poss[0] * M1 + curr_mass_poss[2] * M2) / (M1 + M2)
    y_cm = (curr_mass_poss[1] * M1 + curr_mass_poss[3] * M2) / (M1 + M2)
    return np.array([x_cm, y_cm])

def loadConfigFromCSV(fileName, fileNameTag, verbosity):
    """
    Reads text file formated as csv with: key, value

    :param path: Path of the text file
    :param textFile: Name of the text file
    :return: configDict
     """
    directoryConfig = dir_path

    fileName = fileName + fileNameTag
    dir_and_fileName = path.join(directoryConfig, fileName)

    reader = csv.reader(open(dir_and_fileName))
    configDict = {}
    for row in reader:
        if len(row) != 0:
            if row[0][0] != '#':
                key = row[0]
                if key in configDict:  # In case duplicate
                    pass
                try:
                    configDict[key] = float(row[1])
                except:
                    configDict[key] = row[1].strip()
    if verbosity > 0:
        print_config(configDict)
    return configDict


def print_config(config):
    print(f'{config["tag"]} configurations: ')
    for key in config:
        print('{:>30}: {:>30}'.format(key, str(config[key])))
    print('\n')
    return