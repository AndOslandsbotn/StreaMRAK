import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split


def non_uni_dist(n_tr, cut_off_thr):
    x1 = rng.gamma(shape=1, scale=2, size=n_tr)
    max_val = max(x1)
    x1 = x1/max(x1)

    x2 = np.random.uniform(0, cut_off_thr, size=int(n_tr/3))
    return np.concatenate((x1, x2))

def varying_sinus(x):
    return np.sin(1/(x+0.01)) + np.random.normal(0, 10**(-4), size=len(x))

rng = np.random.default_rng()

save = True
n_tr = 1650000
n_ts = 130000
X_train = non_uni_dist(n_tr, cut_off_thr=1)
y_train = varying_sinus(X_train)

X_test = non_uni_dist(n_ts, cut_off_thr=0.8)
y_test = varying_sinus(X_test)

fig_hist_dist = plt.figure()
count, bins, ignored = plt.hist(X_train, bins=1000)
plt.show()

plt.figure()
plt.plot(X_train, y_train, linestyle='None', marker='s', markersize='0.5', label='Training data')
plt.show()

plt.figure()
plt.plot(X_test, y_test, linestyle='None', marker='o', markersize='2', label='Test data')
plt.show()

### Store the data
dataDir = os.path.join(os.getcwd(), 'VarSinus')

#Training data
path_x_tr = os.path.join(dataDir, 'x_tr')
if save:
    print(f"saving training data: shape {X_train.shape}")
    np.savetxt(path_x_tr, X_train, delimiter=',')
    path_y_tr = os.path.join(dataDir, 'y_tr')
    np.savetxt(path_y_tr, y_train, delimiter=',')

# Test data
path_x_ts = os.path.join(dataDir, 'x_ts')
if save:
    np.savetxt(path_x_ts, X_test, delimiter=',')
path_y_ts = os.path.join(dataDir, 'y_ts')
if save:
    np.savetxt(path_y_ts, y_test, delimiter=',')

# Figure
filename = os.path.join(dataDir, 'histo_sample_dist')
if save:
    fig_hist_dist.savefig(filename)