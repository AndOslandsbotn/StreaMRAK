import numpy as np

def choose_kernel(kernelType = "gaussianKernel"):
    if kernelType == "gaussianKernel":
        return gaussianKernel()
    else:
        print("No such kernel is implemented yet")

class gaussianKernel():
    def __init__(self):
        return

    def calcKernel(self,X1, X2, bandwidth, factor=None):
        n, d = X1.shape

        if factor != None:
            bandwidth = factor * bandwidth

        D = self.broadcastL2Norm(X1, X2)
        D = (-1 / (2 * bandwidth ** 2)) * D
        return np.exp(D)

    def broadcastL2Norm(self,X1, X2):
        X1 = np.expand_dims(X1, axis=2)
        X2 = np.expand_dims(X2.T, axis=0)

        D = np.linalg.norm(X1 - X2, ord=2, axis=1) ** 2
        return D

    def nbTranspose(self, x):
        x_T = x.transpose()
        return x_T