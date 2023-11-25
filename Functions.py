import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

testFunctions = pd.read_csv('Test_functions_0.csv')

# Many Local Minima
def ACKLEY(xx, a=20, b=0.2, c=2*np.pi):
    d = len(xx)
    
    sum1 = 0
    sum2 = 0
    for xi in xx:
        sum1 += xi**2
        sum2 += np.cos(c * xi)
    
    term1 = -a * np.exp(-b * np.sqrt(sum1 / d))
    term2 = -np.exp(sum2 / d)
    
    y = term1 + term2 + a + np.exp(1)
    
    return y

def BUKIN(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = 100 * np.sqrt(np.abs(x2 - 0.01 * x1**2))
    term2 = 0.01 * np.abs(x1 + 10)
    
    y = term1 + term2
    return y

def CROSS_IN_TRAY(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    fact1 = np.sin(x1) * np.sin(x2)
    fact2 = np.exp(np.abs(100 - np.sqrt(x1**2 + x2**2) / np.pi))
    
    y = -0.0001 * (np.abs(fact1 * fact2) + 1)**0.1
    return y

def DROP_WAVE(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    frac1 = 1 + np.cos(12 * np.sqrt(x1**2 + x2**2))
    frac2 = 0.5 * (x1**2 + x2**2) + 2
    
    y = -frac1 / frac2
    return y

def EGGHOLDER(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    term1 = -(x2 + 47) * np.sin(np.sqrt(np.abs(x2 + x1 / 2 + 47)))
    term2 = -x1 * np.sin(np.sqrt(np.abs(x1 - (x2 + 47))))
    
    y = term1 + term2
    return y

def GRIEWANK(xx):
    d = len(xx)
    _sum = 0
    _prod = 1
    
    for ii in range(d):
        xi = xx[ii]
        _sum += xi**2 / 4000
        _prod *= np.cos(xi / np.sqrt(ii + 1))
    
    y = _sum - _prod + 1
    return y

def HOLDER_TABLE(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    fact1 = np.sin(x1) * np.cos(x2)
    fact2 = np.exp(np.abs(1 - np.sqrt(x1**2 + x2**2) / np.pi))
    
    y = -np.abs(fact1 * fact2)
    return y

def Langermann(xx, m=5, c=None, A=None):
    d = len(xx)
    A = np.array([[3, 5], [5, 2], [2, 1], [1, 4], [7, 9]])
    c = np.array([1, 2, 5, 2, 3])
    
    outer = 0
    for ii in range(m):
        inner = 0
        for jj in range(d):
            xj = xx[jj]
            Aij = A[ii, jj]
            inner += (xj - Aij)**2
        new = c[ii] * np.exp(-inner/np.pi) * np.cos(np.pi * inner)
        outer += new

    return outer

def Levy(xx):
    d = len(xx)
    
    w = np.zeros(d)
    for ii in range(d):
        w[ii] = 1 + (xx[ii] - 1) / 4
    
    term1 = (np.sin(np.pi * w[0]))**2
    term3 = (w[d - 1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[d - 1]))**2)
    
    _sum = 0
    for ii in range(d - 1):
        wi = w[ii]
        new = (wi - 1)**2 * (1 + 10 * (np.sin(np.pi * wi + 1))**2)
        _sum += new
    
    y = term1 + _sum + term3
    return y

def Rastrigin(xx):
    d = len(xx)
    _sum = 0
    for xi in xx:
        _sum += (xi**2 - 10 * np.cos(2 * np.pi * xi))
    
    y = 10 * d + _sum
    return y

def schaffer2(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    fact1 = (np.sin(x1**2 - x2**2))**2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2))**2
    
    y = 0.5 + fact1 / fact2
    return y

def schaffer4(xx):
    x1 = xx[0]
    x2 = xx[1]
    
    fact1 = (np.cos(np.sin(np.abs(x1**2 - x2**2))))**2 - 0.5
    fact2 = (1 + 0.001 * (x1**2 + x2**2))**2
    
    y = 0.5 + fact1 / fact2
    return y

def Schwefel(xx):
    d = len(xx)
    _sum = 0
    for xi in xx:
        _sum += xi * np.sin(np.sqrt(np.abs(xi)))
    
    y = 418.9829 * d - _sum
    return y

def Shubert(xx):
    x1 = xx[0]
    x2 = xx[1]
    sum1 = 0
    sum2 = 0

    for ii in range(1, 6):
        new1 = ii * np.cos((ii + 1) * x1 + ii)
        new2 = ii * np.cos((ii + 1) * x2 + ii)
        sum1 += new1
        sum2 += new2

    y = sum1 * sum2
    return y

# Bowl-Shaped
def Bohachevsky(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = x1**2
    term2 = 2 * x2**2
    term3 = -0.3 * np.cos(3 * np.pi * x1)
    term4 = -0.4 * np.cos(4 * np.pi * x2)

    y = term1 + term2 + term3 + term4 + 0.7
    return y

def Perm(xx, b=10):
    d = len(xx)
    outer = 0

    for ii in range(1, d+1):
        inner = 0
        for jj in range(1, d+1):
            xj = xx[jj-1]
            inner += (jj + b) * (xj**ii - (1/jj)**ii)
        outer += inner**2

    y = outer
    return y

def Rotated_Hyper_Ellipsoid(xx):
    d = len(xx)
    outer = 0

    for ii in range(1, d+1):
        inner = 0
        for jj in range(1, ii+1):
            xj = xx[jj-1]
            inner += xj**2
        outer += inner

    y = outer
    return y

def spheref(xx):
    d = len(xx)
    sum_value = 0
    for xi in xx:
        sum_value += xi**2
    y = sum_value
    return y

def sumpow(xx):
    d = len(xx)
    sum_value = 0
    for ii, xi in enumerate(xx):
        new_value = abs(xi)**(ii + 1)
        sum_value += new_value
    y = sum_value
    return y

def sumsqu(xx):
    d = len(xx)
    sum_value = 0
    for ii, xi in enumerate(xx):
        new_value = ii * xi**2
        sum_value += new_value
    y = sum_value
    return y

def trid(xx):
    d = len(xx)
    sum1 = (xx[0] - 1)**2
    sum2 = 0

    for ii in range(1, d):
        xi = xx[ii]
        xold = xx[ii - 1]
        sum1 = sum1 + (xi - 1)**2
        sum2 = sum2 + xi * xold

    y = sum1 - sum2
    return y

def booth(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = (x1 + 2 * x2 - 7)**2
    term2 = (2 * x1 + x2 - 5)**2

    y = term1 + term2
    return y

def matya(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = 0.26 * (x1**2 + x2**2)
    term2 = -0.48 * x1 * x2

    y = term1 + term2
    return y

def mccorm(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = np.sin(x1 + x2)
    term2 = (x1 - x2)**2
    term3 = -1.5 * x1
    term4 = 2.5 * x2

    y = term1 + term2 + term3 + term4 + 1
    return y

def powersum(xx, b=None):
    d = len(xx)

    if b is None:
        if d == 4:
            b = [8, 18, 44, 114]
        else:
            raise ValueError("Value of the d-dimensional vector b is required.")

    outer = 0

    for ii in range(1, d + 1):
        inner = 0
        for jj in range(1, d + 1):
            xj = xx[jj - 1]
            inner += xj**ii
        outer += (inner - b[ii - 1])**2

    y = outer
    return y

def zakharov(xx):
    d = len(xx)
    sum1 = 0
    sum2 = 0

    for ii in range(d):
        xi = xx[ii]
        sum1 += xi**2
        sum2 += 0.5 * (ii + 1) * xi

    y = sum1 + sum2**2 + sum2**4
    return y

# Valley-Shaped
def camel3(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = 2 * x1**2
    term2 = -1.05 * x1**4
    term3 = x1**6 / 6
    term4 = x1 * x2
    term5 = x2**2

    y = term1 + term2 + term3 + term4 + term5
    return y

def camel6(xx):
    x1 = xx[0]
    x2 = xx[1]

    term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
    term2 = x1 * x2
    term3 = (-4 + 4 * x2**2) * x2**2

    y = term1 + term2 + term3
    return y

def dixonpr(xx):
    x1 = xx[0]
    d = len(xx)
    term1 = (x1 - 1)**2

    total_sum = 0
    for ii in range(1, d):
        xi = xx[ii]
        xold = xx[ii - 1]
        new = ii * (2 * xi**2 - xold)**2
        total_sum += new

    y = term1 + total_sum
    return y

def rosen(xx):
    d = len(xx)
    total_sum = 0
    for ii in range(1, d):
        xi = xx[ii - 1]
        xnext = xx[ii]
        new = 100 * (xnext - xi**2)**2 + (xi - 1)**2
        total_sum += new

    y = total_sum
    return y

# Steep Ridges/Drops
def dejong5(xx):
    x1 = xx[0]
    x2 = xx[1]
    total_sum = 0

    A = np.zeros((2, 25))
    a = np.array([-32, -16, 0, 16, 32])
    A[0, :] = np.tile(a, 5)
    ar = np.tile(a, 5)
    ar = ar.reshape(5, 5).T.flatten()
    A[1, :] = ar

    for ii in range(25):
        a1i = A[0, ii]
        a2i = A[1, ii]
        term1 = ii + 1
        term2 = (x1 - a1i)**6
        term3 = (x2 - a2i)**6
        new = 1 / (term1 + term2 + term3)
        total_sum += new

    y = 1 / (0.002 + total_sum)
    return y

def easom(xx):
    x1 = xx[0]
    x2 = xx[1]

    fact1 = -np.cos(x1) * np.cos(x2)
    fact2 = np.exp(-((x1 - np.pi)**2 + (x2 - np.pi)**2))

    y = fact1 * fact2
    return y

def michal(xx, m=10):
    d = len(xx)
    total_sum = 0

    for ii in range(d):
        xi = xx[ii]
        new = np.sin(xi) * (np.sin(ii * xi**2 / np.pi))**(2 * m)
        total_sum += new

    y = -total_sum
    return y

def RosenbrockConstrained1(x):
    if ((x[0] -1)**3 - x[1] + 1) <= 0 and (x[0] + x[1]-2) <= 0:
        return (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
    else:
        return float('Inf')

def RosenbrockConstrained2(x):
    if (x[0]**2 + x[1]**2) <=2:
        return (1-x[0])**2 + 100 * (x[1] - x[0]**2)**2
    else:
        return float('Inf')

def Mishras_Bird(x):
    if ((x[0] + 5)**2 + (x[1] + 5)**2) < 25:
        return np.sin(x[0])*(np.exp(1-np.cos(x[1]))**2)+np.cos(x[1])*(np.exp(1-np.sin(x[0]))**2)+(x[0]-x[1])**2
    else:
        return float('Inf')

def Townsend(x):
    t = np.arctan2(x[0], x[1])
    if (x[0]**2 + x[1]**2) < (2 * np.cos(t) - 1/2 * np.cos(2 * t)- 1/4 * np.cos(3 * t) - 1/8 * np.cos(4 * t))**2\
          + (2 * np.sin(t))**2:
        return -1 * (np.cos((x[0]-0.1) * x[1]))**2 - x[0] * np.sin(3 * x[0] + x[1])
    else:
        return float('inf')

def Gomez_and_Levy(x):
    if (-np.sin(4 * np.pi * x[0]) + 2 * (np.sin(2 * np.pi * x[1]))**2) <= 1.5:
        return 4 * x[0]**2 - 2.1 * x[0]**2 + 1/3 * x[0]**6 + x[0] * x[1] - 4 * x[1]**2 + 4 * x[1]**4
    else:
        return float('inf')

def Simionescu(x):
    if (x[0]**2 + x[1]**2) <= (1 + 0.2 * np.cos(8 * np.arctan(x[0]/x[1])))**2:
        return 0.1 * x[0] * x[1]
    else:
        return float('inf')
    
def CEC(x, fn = 1, dim = 10):
    try:
        bench.init(fn, dim) # Init function 3 with dimension 10
    except Exception as e:
        from cec2019comp100digit import cec2019comp100digit
        bench = cec2019comp100digit
        bench.init(fn, dim) # Init function 3 with dimension 10
    return bench.eval(x)


def bent_cigar_function(x):
    return x[0]**2 + 1e6 * x[1]**2 + np.sum(x[2:]**2)

def Discus(x):
    return 1e6 * x[0]**2 + np.sum(x[1:]**2)

def six_hump(x):
    term1 = (4 - 2.1 * x[0]**2 + x[0]**4 / 3) * x[0]**2
    term2 = x[0] * x[1]
    term3 = (-4 + 4 * x[1]**2) * x[1]**2

    return term1 + term2 + term3