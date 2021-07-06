import scipy.linalg as lalg
import numpy as np

from StreaMRAK.StreaMRAKutil.choose_kernel import choose_kernel

class FALKON_matrixSystem():
    """This class implements the matrix system for the FALKON
    algorithm."""
    def __init__(self, config):
        self.isGPU = config['useGPU']
        self.kernelType = config['kernelType']
        self.kernel_obj = choose_kernel(self.kernelType)

    def calcW(self, cholT, cholTt, cholA, cholAt, KnmTKnm, n, v):
        """
        With the preconditioner B = 1/sqrt(n)*invers(T)*invers(A) we have
        W = transpose(B)*H*B.
        :param cholT(v): Solves Tx=v for x
        :param cholTt(v): Solves T'x=v for x
        :param cholA(v): Solves Ax=v for x
        :param cholAt(v): Solves A'x=v for x
        :param KnmTKnm: m x m matrix
        :param n: number of training points
        :param v: parameter to apply the matrix W on
        :return:
        """
        v = cholA(v)
        KnmTKnmBv = np.dot(KnmTKnm, cholT(v))
        W = (1 / n) * cholAt(cholTt(KnmTKnmBv) + n*self.reg_param * v)
        return W

    def calcKmm(self, landmarks, scale):
        return self.kernel_obj.calcKernel(landmarks, landmarks, scale)

    def calcKnm(self, x_tr, landmarks, scale):
        return self.kernel_obj.calcKernel(x_tr, landmarks, scale)

    def calcKnmTKnm(self, x_tr, landmarks, scale):
        Knm = self.calcKnm(x_tr, landmarks, scale)
        Knm_T = self.nbTranspose(Knm)
        return np.dot(Knm_T, Knm)

    def calcZm(self, x_tr, y_tr, landmarks, scale):
        Knm = self.calcKnm(x_tr, landmarks, scale)
        Knm_T = self.nbTranspose(Knm)
        return np.dot(Knm_T, y_tr)

    #@nb.jit(nopython=True)
    def nbTranspose(self, X):
        X_T = X.transpose()
        return X_T

class FALKON_precond():
    """
    This class implements the preconditioner functionality
    for the FALKON algorithm
    """
    def __init__(self, config):
        self.reg_param = config['reg_param']
        self.chol_reg = config['chol_reg']
        return

    def create_AandT(self, Kmm):
        """
        This functions creates the matrices T, Tt, A, At which are
        used to define the preconditioner B = 1/sqrt(n)*invers(T)*invers(A)
        where T = Cholesky(Kmm) and A = Cholesky((1/m)*T*T.transpose + lambda*I)
        :param Kmm: Matrix
        :return: T, Tt, A, At
        """
        # T matrix and T.transpose matrix
        # self.debug(Kmm)
        m, _ = Kmm.shape
        T = lalg.cholesky(Kmm+self.chol_reg*np.identity(m))
        Tt = self.nbTranspose(T)

        # A matrix and A.transpose matrix
        inter_med = (1/m)*np.dot(T, Tt) + self.reg_param * np.identity(m)
        A = lalg.cholesky(inter_med)

        return T, A

    def create_chol_solvers(self, Kmm):
        """
        :param Kmm: Matrix
        :return: Return handles to lambda functions
        CholR(x): handle to linear solver for T*u= x
        CholRt(x): handle to linear solver for T'*u= x
        CholA(x): handle to linear solver for A*u= x
        CholAt(x): handle to linear solver for A'*u= x
        """

        T, A = self.create_AandT(Kmm)

        cholT = self.solve_triangular_system(T, systemType='N')
        cholTt = self.solve_triangular_system(T, systemType='T')
        cholA = self.solve_triangular_system(A, systemType='N')
        cholAt = self.solve_triangular_system(A, systemType='T')

        return cholT, cholTt, cholA, cholAt

    def solve_triangular_system(self, matrix, systemType):
        return lambda x: lalg.solve_triangular(matrix, x, trans=systemType)

    # @nb.jit(nopython=True)
    def nbTranspose(self, X):
        X_T = X.transpose()
        return X_T


class FALKON_conjgrad():
    def __init__(self, config):
        self.conj_grad_max_iter = config['conj_grad_max_iter']
        self.conj_grad_thr = config['conj_grad_thr']
        return

    def conjgrad(self, operatorW, b, x):
        """
        A function to solve W\beta = b, with the
        conjugate gradient method. See also
        wikipedia: https://en.wikipedia.org/wiki/Conjugate_gradient_method

        :param operatorW: an operator matrix, real symmetric positive definite
        :param b: vector, right hand side vector of the system
        :param x: vector, starting guess for \beta
        :return: x which is the estimated value of beta
        """

        if x.size == 0:
            res = b
            if b.ndim > 1:
                print("We have more than 2 dim")
                m_b, dim_b = b.shape
                x = np.zeros((m_b, dim_b))
            else:
                m_b = b.shape
                x = np.zeros((m_b))
        else:
            res = b - operatorW(x)

        p = res
        res_old = np.dot(np.transpose(res), res)
        for i in range(int(self.conj_grad_max_iter)):
            Ap = operatorW(p)
            step = (res_old / np.dot(np.transpose(p), Ap))
            x = x + np.dot(p, step)
            res = res - np.dot(Ap, step)
            res_new = np.dot(np.transpose(res), res)
            if np.sqrt(res_new) < self.conj_grad_thr:
                break
            p = res + (res_new / res_old) * p
            res_old = res_new
        return x

class FALKON_Solver(FALKON_precond, FALKON_conjgrad, FALKON_matrixSystem):
    """This class (fits the kernel model) i.e.
    (solves the for the alphas {\alpha^{(s)}_i}),
    at scale s, in the model function
    f(x) = \Sum_{i=1}^m k_s(x,x_i)\alpha^{(s)}_i,
    where s is the scale
    """
    def __init__(self, config):
        FALKON_precond.__init__(self, config)
        FALKON_conjgrad.__init__(self, config)
        FALKON_matrixSystem.__init__(self, config)

        self.kernelType = config['kernelType']
        self.reg_param = config['reg_param']
        self.lm_set = False

    def fit_at_scale(self, Kmm, KnmTKnm, Zm, n_tr):
        """
        This function solves the linear system Bt*H*B*\beta = Bt*Knmt*y
        for \beta, and the get \alpha  from \alpha = B\beta.
        :param Kmm: System matrix size, m x m
        :param KnmTKnm: System matrix size, m x m
        :param Zm: System vector size, m
        :param n_tr: Number of training points used to form KnmTKnm and Zm
        :return: alpha, coefficients in the kernel model
        """

        # Get the preconditioner matrices as lambda functions, that solve
        # a triangular linear system
        cholT, cholTt, cholA, cholAt = self.create_chol_solvers(Kmm)

        # Compute right hand side of W*beta = b, where W = Bt*H*B and b =Bt*Zm
        b = cholAt(cholTt(Zm))

        # Compute the system matrix W as a linear operator that
        # acts on v
        W = lambda v: self.calcW(cholT, cholTt, cholA, cholAt, KnmTKnm, n_tr, v)

        # Perform conjugate gradient
        _, yd = b.shape
        beta = np.zeros_like(b)

        for i in range(0, yd):
            initialGuessBeta = np.array([])
            beta[:, i] = self.conjgrad(W, b[:, i], initialGuessBeta)
        # Get alpha
        alpha = cholT(cholA(beta))
        return alpha

    def predict_at_scale(self, X, landmarks, alpha, scale):
        Knm = self.calcKnm(X, landmarks, scale)
        return np.dot(Knm, alpha)
