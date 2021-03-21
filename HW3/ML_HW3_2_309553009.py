import numpy as np
import math
from argparse import ArgumentParser

def GaussianDataGen(mean, var):
    # box-muller
    u = np.random.uniform(0.0, 1.0)
    v = np.random.uniform(0.0, 1.0)
    z = np.sqrt(-2.0 * np.log(u)) * np.sin(2 * np.pi * v)
    return mean + np.sqrt(var) * z

def SequentialEstimator(mean=0.0, var=0.0, m2n=0.0, n=0):
    data_pt = GaussianDataGen( args.input_mean, args.input_var)
    print('Add data point: {}'.format(data_pt))
    n += 1
    # Welford's online algorithm
    new_mean = mean + (data_pt - mean) / n
    new_m2n = m2n + (data_pt - mean) * (data_pt - new_mean)
    new_var = new_m2n / (n-1) if(n != 1) else 0.0
    # new_var = (n-2)*var/(n-1) + (data_pt - mean)**2/n if(n != 1) else 0.0     # instable
    print('Mean = {:.15f}    Variance = {:.15f}'.format(new_mean, new_var))
    return (new_mean, new_var, new_m2n, n)

################################ main ###################################

### 2
parser = ArgumentParser()
parser.add_argument('--input_mean', type=float, default=3.0)
parser.add_argument('--input_var', type=float, default=5.0)
args = parser.parse_args()
# python hw3_2.py --input_mean 3.0 --input_var 5.0
# import time
# start = time.time()

print('Data point source function: N({}, {})\n'.format(
    args.input_mean, args.input_var))

(new_mean, new_m2n, new_var, n) = (0.0, 0.0, 0.0, 0)
while(1):
    (mean, var, m2n) = (new_mean, new_var, new_m2n)
    new_mean, new_var, new_m2n, n = SequentialEstimator(mean, var, m2n, n)
    if(abs(new_mean - mean) < 1e-4 and abs(new_var - var) < 1e-4):
        break   # converge

# print(time.time() - start)    # 170