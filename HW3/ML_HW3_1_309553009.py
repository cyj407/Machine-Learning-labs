import numpy as np

def GaussianDataGen(mean, var):
    # box-muller
    u = np.random.uniform(0.0, 1.0)
    v = np.random.uniform(0.0, 1.0)
    z = np.sqrt(-2.0 * np.log(u)) * np.sin(2 * np.pi * v)
    return mean + np.sqrt(var) * z

def PolynomialDataGen(n, a, w):
    x = np.random.uniform(-1.0, 1.0)
    X = np.array([(x ** deg)  for deg in range(n)]).reshape(-1)
    W = np.array(w).reshape(-1)
    e = GaussianDataGen(0, a)
    return x, np.dot( np.transpose(W), X) + e

################################ main ###################################

### 1
print('Gaussian Data Generator')
m = float(input('mean = '))
s = float(input('variance = '))
print(GaussianDataGen(m, s))
print()

print('Polynomial Basis Linear Model Data Generator')
n = int(input('basis number n = '))
a = float(input('a = '))
print('w ({}x1 vector) :'.format(n))
w = [float(input('  w{} = '.format(i))) for i in range(n)]
print(PolynomialDataGen(n, a, w))