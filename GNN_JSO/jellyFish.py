import numpy as np


# Step 1: Initialize population of jellyfish Xi using logistic chaotic map
def logistic_map(r, x, n):
    for _ in range(n):
        #----------- eqn 1
        x = r * x * (1 - x)
    return x


# Step 2: Calculate the quantity of food at each Xi as f(Xi)
def food_quantity(x):
    # Define your food quantity function here
    return x ** 2  # Example function, replace with actual implementation


# Step 3: Find the jellyfish at the location currently with the most food (X*)
def find_best_location(population):
    best_idx = np.argmax([food_quantity(x) for x in population])
    return population[best_idx]

def opt():
    # Parameters
    npop = 100  # Population size
    MaxIter = 100  # Maximum number of iterations
    C0 = 0.5  # Threshold for ocean current
    lb = 0  # Lower bound for boundary conditions
    ub = 1  # Upper bound for boundary conditions
    r = 3.9  # Control parameter for logistic map
    x0 = 0.1  # Initial condition for logistic map
    beta = 0.1  # Parameter for ocean current calculation
    mu = 0.1  # Parameter for ocean current calculation
    lamda = 0.1  # Parameter for movement inside swarm

    # Step 1: Initialize population of jellyfish Xi
    population = [logistic_map(r, x0, np.random.randint(100)) for _ in range(npop)]

    # Step 3: Find the jellyfish at the location currently with the most food (X*)
    X_star = find_best_location(population)

    # Step 4: Main optimization loop
    best_solution = None
    best_fitness = float('-inf')

    for t in range(1, MaxIter + 1):
        for i in range(npop):
            # Step 6: Calculate the time control c(t)
            c_t = np.abs((1 - t / MaxIter) * (2 * np.random.rand() - 1))

            # Step 7: Jellyfish follows ocean current
            if c_t > C0:
                trend = X_star - beta * np.random.rand() * mu

                ##-------- eqn 2
                population[i] = population[i] + np.random.rand() * trend
            # Step 8: Jellyfish moves inside a swarm
            elif np.random.rand() > (1 - c_t):
                population[i] = population[i] + lamda * np.random.rand() * (ub - lb)
            # Step 9: Jellyfish explores
            else:
                j = np.random.choice([idx for idx in range(npop) if idx != i])
                direction = population[j] - population[i] if food_quantity(population[i]) >= food_quantity(
                    population[j]) else population[i] - population[j]
                step = np.random.rand() * direction
                population[i] = population[i] + step

            # Step 11: Check boundary conditions
            if population[i] < lb:
                population[i] = lb
            elif population[i] > ub:
                population[i] = ub

            # Step 12: Calculate the quantity of food at new location
            food_at_new_location = food_quantity(population[i])

            # Update best solution so far
            if food_at_new_location > best_fitness:
                best_solution = population[i]
                best_fitness = food_at_new_location

    return best_fitness


import tensorflow as tf
# ---------- thin Keras wrapper: lets you use your algm() in model.compile ----------
class JSO(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.001, name="JSO", **kwargs):
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

