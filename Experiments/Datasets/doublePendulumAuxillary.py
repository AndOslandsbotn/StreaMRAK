from math import acos, cos
import numpy as np

def calcEnergy(state, configDict):
    the1, the2, w1, w2 = state
    g = configDict['G']
    l1 = configDict['L1']
    l2 = configDict['L2']
    m1 = configDict['M1']
    m2 = configDict['M2']

    # Kinteic energy
    E_k_1 = (m1 / 2) * (l1 * w1) ** 2
    E_k_2 = (m2 / 2) * ((l1 * w1) ** 2 + (l2 * w2) ** 2 + 2 * l1 * l2 * w1 * w2 * cos(the2 - the1))
    E_k = E_k_1 + E_k_2

    # Potential energy
    E_p_1 = -m1 * g * l1 * cos(the1)
    E_p_2 = -m2 * g * (l1 * cos(the1) + l2 * cos(the2))
    E_p = E_p_1 + E_p_2

    E = E_p + E_k
    return E

def calcConstEnergyDomainBoundaries(E_I, configDict):
    g = configDict['G']
    l1 = configDict['L1']
    l2 = configDict['L2']
    m1 = configDict['M1']
    m2 = configDict['M2']

    th1_max = np.radians(180) #acos(-(E_I + m2 * g * l2) / ((m1 + m2) * l1 * g))
    print(np.degrees(th1_max))
    print("(E_I+(m1+m2)*g*l1)/(m2*g*l2) ", (E_I + (m1 + m2) * g * l1) / (m2 * g * l2))
    th2_max = np.radians(180) #acos(-(E_I + (m1 + m2) * g * l1) / (m2 * g * l2))

    w1_max = np.sqrt(2 * (E_I + ((m1 + m2) * l1 + m2 * g) * g) / (l1 ** 2 * (m1 + m2)))
    w2_max = np.sqrt(2 * (E_I + ((m1 + m2) * l1 + m2 * l2) * g) / (m2 * l2 ** 2))

    return th1_max, th2_max, w1_max, w2_max