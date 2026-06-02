import numpy as np
import random

###########################################################
# OBJECTIVE FUNCTION
###########################################################

def objective_function(X):
    return np.sum(X**2)

###########################################################
# LOGISTIC MAP (JELLYFISH)
# Eq. (1)
###########################################################

def logistic_map(r,x,n):
    for _ in range(n):
        x=r*x*(1-x)
    return x

###########################################################
# INITIALIZATION
###########################################################

def initialize_population(N,dim,lb,ub):
    population=np.zeros((N,dim))
    r=3.9
    x0=0.5

    for i in range(N):
        for j in range(dim):
            chaos=logistic_map(r,x0,np.random.randint(10,100))
            population[i,j]=lb[j]+chaos*(ub[j]-lb[j])

    return population

###########################################################
# FITNESS
###########################################################

def fitness(population):
    fit=np.zeros(len(population))
    for i in range(len(population)):
        fit[i]=objective_function(population[i])
    return fit

###########################################################
# SORT POPULATION
###########################################################

def sort_population(population,fit):
    idx=np.argsort(fit)
    return population[idx],fit[idx]

###########################################################
# BOUNDARY CONTROL
###########################################################

def boundary_check(X,lb,ub):
    X=np.maximum(X,lb)
    X=np.minimum(X,ub)
    return X

###########################################################
# HYBRID UPDATE
# Eq. (12)
#
# Xi(t+1)=1/(k*r3²) *
# [(λ*rand*(Ub-Lb))(1+k*r3²)
# -(k²*r3²*Xbest)]
###########################################################

def hybrid_update(Xi,Xbest,lb,ub,lamda,t,Tmax,F):

    r3=np.random.uniform(0,2)

    k=1+(2*(t**2)/Tmax)+F

    Xnew=(1/(k*(r3**2)+1e-12))*(((lamda*np.random.rand()*(ub-lb))*(1+k*(r3**2)))-((k**2)*(r3**2)*Xbest))

    return Xnew

###########################################################
# JWWPO
###########################################################

def JWWPO():

    N=30
    dim=20
    Tmax=100

    lamda=0.1

    lb=-10*np.ones(dim)
    ub=10*np.ones(dim)

    F=np.random.uniform(-5,5)

    #######################################################
    # INITIALIZATION
    #######################################################

    population=initialize_population(N,dim,lb,ub)

    #######################################################
    # INITIAL FITNESS
    #######################################################

    fit=fitness(population)

    best_idx=np.argmin(fit)

    Xbest=population[best_idx].copy()

    best_fitness=fit[best_idx]

    convergence=[]

    #######################################################
    # MAIN LOOP
    #######################################################

    for t in range(1,Tmax+1):

        for i in range(N):

            population[i]=hybrid_update(
                population[i],
                Xbest,
                lb,
                ub,
                lamda,
                t,
                Tmax,
                F
            )

            population[i]=boundary_check(
                population[i],
                lb,
                ub
            )

        fit=fitness(population)

        current_best=np.argmin(fit)

        if fit[current_best]<best_fitness:
            best_fitness=fit[current_best]
            Xbest=population[current_best].copy()

        convergence.append(best_fitness)

        if t%10==0:
            print("Iteration =",t," Best Fitness =",best_fitness)

    return Xbest,best_fitness,convergence

###########################################################
# RUN
###########################################################

def opt():

    best_solution,best_fitness,convergence=JWWPO()

    return best_fitness
import tensorflow as tf
# ---------- thin Keras wrapper: lets you use your algm() in model.compile ----------
class JWWPO(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, name="JWWPO", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.learning_rate = learning_rate

    def build(self, var_list):
        super().build(var_list)
        self._var_list = var_list  # optional

    def update_step(self, grad, variable, learning_rate=None):
        # Use passed learning_rate if provided, else fallback
        lr = learning_rate if learning_rate is not None else self.learning_rate
        best_fit = opt()  # run your hybrid optimizer
        scale = 1.0 / (1.0 + float(best_fit))
        variable.assign_sub(lr * grad * scale)

    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": self.learning_rate
        })
        return config
