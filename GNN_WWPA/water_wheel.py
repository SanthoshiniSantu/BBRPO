import random
import time
import numpy as np

rng = np.random.default_rng()


# Sort ObjFun.
def SortObjFun(Fit):
    ObjFun = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return ObjFun, index


# Sort the position  according to ObjFun.
def select_prey(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew


def fun(X):
    output = sum(np.square(X))
    return output


def fit(N, X):
    ObjFun = np.zeros([N, 1])
    for i in range(N):
        ObjFun[i] = CaculateObjFun1(X[i, :], fun)  ###### eqn 3
    return ObjFun


# This function is to initialize
def initial(N, dim, ub, lb):
    X = np.zeros([N, dim])
    for i in range(N):
        for j in range(dim):
            X[i, j] = lb[j] + random.random() * (ub[j] - lb[j])  ###### eqn 2
    return X


# Calculate ObjFun values
def CaculateObjFun1(X, fun):
    ObjFun = fun(X)
    return ObjFun


def WW():
    N = 10
    t, T = 1, 10
    M = 10  # The dimension.
    fl = -10  # The lower bound of the search interval.
    ul = 10  # The upper bound of the search interval.
    F,C = np.random.uniform(-5,5),np.random.uniform(-5,5)
    lb = fl * np.ones([M, 1])
    ub = ul * np.ones([M, 1])
    X = initial(N, M, lb, ub)  ### eqn 1
    Cand_Prey = np.zeros((N, M))
    ObjFun = fit(N, X)
    Xnew = np.zeros([N, M])


    rand = np.random.rand(N, M)
    best_Pos = np.argmax(X, axis=1)
    best = np.argmax(ObjFun)

    while t < T:


        ObjFun, sortIndex = SortObjFun(ObjFun)  # Sort the ObjFun values
        SCP = select_prey(X, sortIndex)  ###### strongest walrusus
        r1 =r3 = random.uniform(0, 2)

        r2 =K= random.uniform(0, 1)
        I = random.uniform(1, 2)
        r = random.uniform(0, 1)
        mu = 100
        sigma = 50

        for i in range(N-1):
            ######### phase 1 :::::: Position Identification and Hunting of Insects
            if r <0.5:
                W = r1 *(X[i,:]+2*K)   ########eqn 4
                Xnew[i+1,:] = X[i,:] + W *(2*K +r2)  ##### eqn 5

                if (Xnew[i + 1, :] >X[i + 1, :]).any() ==True:
                    Xnew[i + 1, :] = random.gauss(mu, sigma) + r1 *(X[i,:] +2*K /W)  ###### eqn 6
            else:
                ####### phase 2:::::::::::: Carrying the Insect in the Suitable Tube
                W = r3 * (K* best_Pos[i] + r3* X[i,:])  ########eqn 7
                Xnew[i + 1, :] = X[i, :] + W * K ##### eqn 8

                if (Xnew[i + 1, :] > X[i + 1, :]).any() == True:
                    Xnew[i + 1, :] = (r1 +K) * np.sin(F/C)  ###### eqn 9


            K = (1+ (2*(t**2)/T) +F)  ########eqn 10



            ObjFun_p1 = fit(N, X)
            bst = np.argmax(ObjFun_p1)

            best_Pos[t] =X[t][bst]
        t += 1
    return np.mean(best_Pos)


import tensorflow as tf
# ---------- thin Keras wrapper: lets you use your algm() in model.compile ----------
class WWPA(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, name="WWPA", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.learning_rate = learning_rate

    def build(self, var_list):
        super().build(var_list)
        self._var_list = var_list  # optional

    def update_step(self, grad, variable, learning_rate=None):
        # Use passed learning_rate if provided, else fallback
        lr = learning_rate if learning_rate is not None else self.learning_rate
        best_fit = WW()  # run your hybrid optimizer
        scale = 1.0 / (1.0 + float(best_fit))
        variable.assign_sub(lr * grad * scale)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self.learning_rate
        })
        return config


