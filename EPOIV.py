import numpy as np
from FunctionCall import *

# Generate population function
def generate_population(LB, UB, pop_size, dim):
    return np.random.uniform(LB, UB,(pop_size, dim))

# The temperature profile around the huddle
def temp(Itr, MaxItr, R):
    R = np.random.uniform()
    if R >= 1:
        T = 0
    else:
        T = 1
    return T - Itr/(MaxItr - Itr)

# Fix bounds function
def fix_bounds(x, LB, UB):
    x_lenght = len(x)
    for i in range(x_lenght):
        if x[i] > UB or x[i] < LB:
            x[i] = np.random.uniform(LB, UB)
    return x

# Relocate penguin position
def Relocate(P, P_other, Itr, MaxItr, LB, UB, R, func1, CurEval, thershold = 0.3):
    # Compute the temperature profile around the huddle
    R = np.random.uniform(-1, 1)
    Temp = temp(Itr, MaxItr, R)
    # The polygon grid accuracy
    P_grid = abs(P - P_other)
    # Calculate avoid collision parameter
    A = 1.5 * (Temp + P_grid) * np.random.uniform() - Temp
    # The social forces of emperor penguins
    S = np.sqrt(np.random.uniform(3, 15) * np.exp(-Itr/np.random.uniform(3, 15)) - np.exp(-Itr))
    # Relocate penguin
    D = abs(S * P - np.random.uniform(0, 1, len(P_other)) * P_other)
    modified = fix_bounds(P - A * D, LB, UB)
    newposition = np.array([])
    for i in range(len(modified)):
        if np.random.uniform() > thershold:
            newposition = np.append(newposition,P[i])
        else:
            newposition = np.append(newposition, modified[i])
    Solutions = [P_other, newposition, modified]
    Evals = [func1(i) for i in Solutions]
    newposition = Solutions[np.argmin(Evals)]
    # print(min(Evals), 1)
    return newposition, func1(newposition)

# Emperor penguin optimizer
def EPOIV(LB, UB, dim, pop_size, MaxItr, R, fn, T = 0.4):
    try:
        func1 = func(fn)
        func1(np.random.uniform(-5, 5, 10))
    except Exception as e:
        func1 = lambda x:CEC(x, fn, dim)
    # Initialize the emperor penguins population
    population = generate_population(LB, UB, pop_size, dim)
    # Calculate the fitness value of each search agent
    fitness = [func1(x) for x in population]
    best = min(fitness)
    Gbest = population[np.argmin(fitness),:]
    
    # k = 1
    P_vector_1 = np.array([])
    # P_vector_2 = np.arange(MaxItr)
    # i belongs to [0, MaxItr -1]
    for i in range(MaxItr):
        for j in range(pop_size):
            # Update position
            Current_Position = population[j]
            Relocating = Relocate(Gbest, Current_Position, i, MaxItr, LB, UB, R, func1,fitness[j], T)
            Current_Position = Relocating[0]
            fitness[j] = Relocating[1]
            # print(fitness[j], 2)
            # if Eval_modified <= fitness[j]:
            #     Eval_new_position = Eval_modified
            #     Current_Position = modified
            #     fitness[j] = Eval_modified
            # else:
            #     Eval_new_position = fitness[j]
            # Update the position of the optimal solution
            if fitness[j] <= best:
                best = fitness[j]
                Gbest = np.copy(Current_Position)
        P_vector_1 = np.append(P_vector_1,best)

    return best, Gbest, P_vector_1

# Relocate penguin position
def Relocate_classic(P, P_other, Itr, MaxItr, LB, UB, R):
    # Compute the temperature profile around the huddle
    R = np.random.uniform(0, R)
    Temp = temp(Itr, MaxItr, R)
    # The polygon grid accuracy
    P_grid = abs(P - P_other)
    # Calculate avoid collision parameter
    A = 2 * (Temp + P_grid) * np.random.uniform() - Temp
    # The social forces of emperor penguins
    S = np.sqrt(np.random.uniform(5, 10) * np.exp(-Itr/np.random.uniform(5, 50)) - np.exp(-Itr))
    # Relocate penguin
    D = abs(S * P - np.random.uniform() * P_other)
    return fix_bounds(P - A * D, LB, UB)

# Emperor penguin optimizer
def PCA_classic(LB, UB, dim, pop_size, MaxItr, R, fn):
    try:
        func1 = func(fn)
        func1(np.random.uniform(-5, 5, 10))
    except Exception as e:
        func1 = lambda x:CEC(x, fn, dim)
    # Initialize the emperor penguins population
    population = generate_population(LB, UB, pop_size, dim)
    # Calculate the fitness value of each search agent
    fitness = [func1(x) for x in population]
    Gbest = population[fitness.index(min(fitness)),:]
    best = min(fitness)
    # k = 1
    P_vector_1 = np.array([])
    # P_vector_2 = np.arange(MaxItr)
    for i in range(MaxItr):
        for j in range(pop_size):
            # Update position
            Current_Position = population[j]
            Current_Position = Relocate_classic(Gbest, Current_Position, i, MaxItr, LB, UB, R)
            Eval_new_position = func1(Current_Position)
            # Update the position of the new optimal solution
            if Eval_new_position <= best:
                best = Eval_new_position
                Gbest = Current_Position
        P_vector_1 = np.append(P_vector_1,best)
        

    # plt.plot(P_vector_2, P_vector_1)
    # plt.ylabel('Fitness')
    # plt.xlabel('Iteration')
    return best, Gbest, P_vector_1

# EPO with Levy fligth and Gaussian mutation
def LevyFligth(alpha=1.5, size=1):
    # mu = np.random.normal(loc=0, scale=np.sqrt(3))
    # v = np.random.normal(loc=0, scale=np.sqrt(3))
    # step = mu / (np.abs(v) ** (1/3))
    # return (step + lastStep)
    sigma_u = (np.math.gamma(1 + alpha) * np.sin(np.pi * alpha / 2) /
        (np.math.gamma((1 + alpha) / 2) * alpha * 2**((alpha - 1) / 2)))**(1 / alpha)

    levy = np.random.normal(0, sigma_u, size)
    return levy[0]

def GaussianMutation(value):
    gaussian_value = np.random.uniform() * np.random.normal(loc=0, scale=np.sqrt(0.1))
    return(value + gaussian_value)

def Relocate_LevyGaussian(P, P_other, Itr, MaxItr, LB, UB, R, func1):
    # Compute the temperature profile around the huddle
    R = np.random.uniform(0, R)
    Temp = temp(Itr, MaxItr, R)
    # The polygon grid accuracy
    P_grid = abs(P - P_other)
    # Calculate avoid collision parameter
    A = 2 * (Temp + P_grid) * np.random.uniform() - Temp
    # The social forces of emperor penguins
    S = np.sqrt(np.random.uniform(5, 10) * np.exp(-Itr/np.random.uniform(5, 50)) - np.exp(-Itr))
    # Relocate penguin
    D = abs(S * P - np.random.uniform(0, 1, len(P_other)) * P_other)
    # LevySteps[solNum] = LevyFligth(LevySteps[solNum])
    LevyGaussianMutated = P * LevyFligth() - GaussianMutation(A * D)
    # solutions = [P_other, LevyGaussianMutated]
    # Evals =[func1(i) for i in solutions]
    # bestLocal = solutions[np.argmin(Evals)]
    return fix_bounds(LevyGaussianMutated, LB, UB)
    # return fix_bounds(P  - A * D, LB, UB)

def EPO_LevyGaussian(LB, UB, dim, pop_size, MaxItr, R, fn):
    try:
        func1 = func(fn)
        func1(np.random.uniform(-5, 5, 10))
    except Exception as e:
        func1 = lambda x:CEC(x, fn, dim)
    # Initialize the emperor penguins population
    population = generate_population(LB, UB, pop_size, dim)
    # Calculate the fitness value of each search agent
    fitness = [func1(x) for x in population]
    Gbest = population[fitness.index(min(fitness)),:]
    best = min(fitness)
    # LevySteps = np.zeros(pop_size)
    # k = 1
    P_vector_1 = np.array([])
    # P_vector_2 = np.arange(MaxItr)
    for i in range(MaxItr):
        for j in range(pop_size):
            # Update position
            Current_Position = population[j]
            Current_Position = Relocate_LevyGaussian(Gbest, Current_Position, i, MaxItr, LB, UB, R, func1)
            Eval_new_position = func1(Current_Position)
            # Update the position of the new optimal solution
            if Eval_new_position <= best:
                best = Eval_new_position
                Gbest = np.copy(Current_Position)
        P_vector_1 = np.append(P_vector_1,best)
        

    # plt.plot(P_vector_2, P_vector_1)
    # plt.ylabel('Fitness')
    # plt.xlabel('Iteration')
    return best, Gbest, P_vector_1