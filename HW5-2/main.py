import numpy as np
from scipy.spatial.distance import cdist
from libsvm.svmutil import svm_train, svm_predict
from libsvm.svm import *

def train_and_predict(x, y, kernel, option=''):
    kernel_type = {'linear':'0', 'polynomial':'1', 'rbf':'2', 'sigmoid':'3', 'self-defined':'4'}
    x_train, x_test = x
    y_train, y_test = y
    param = '-q -t ' + kernel_type[kernel] + option  # -q: suppress the output in libsvm
    m = svm_train(y_train, x_train, param) 
    pred, pred_acc, pred_val = svm_predict(y_test, x_test, m)

def gridSearch(x, y, kernel='rbf'):
    kernel_type = {'linear':'0', 'polynomial':'1', 'rbf':'2', 'self-defined':'4'}
    C = [1, 10, 20, 30]
    gamma = [0.2, 0.1, 0.05, 0.01]
    degree = [2, 3, 4, 5]
    coef0 = [0, 5, 10]

    max_acc = 0.0
    opt = '-q -s 0 -v 5 '
    if(kernel == 'linear'):
        best_param = (C[0])
        for c in C:
            # print('C = {}'.format(c))
            acc = svm_train(y, x, opt + '-t {} -c {}'.format(kernel_type[kernel], c))
            if(acc > max_acc):
                max_acc = acc
                best_param = (c)
    elif(kernel == 'polynomial'):
        best_param = (C[0], gamma[0], coef0[0], degree[0])
        for c in C:
            for g in gamma:
                for d in degree:
                    for r in coef0:
                        # print('C = {}, gamma = {}, degree = {}, coef0 = {}'.format(c, g, d, r))
                        acc = svm_train(y, x, opt + '-t {} -c {} -g {} -d {} -r {}'.format(kernel_type[kernel], c, g, d, r))
                        if(acc > max_acc):
                            max_acc = acc
                            best_param = (c, g, d, r)
    elif(kernel == 'rbf'):
        best_param = (C[0], gamma[0])
        for c in C:
            for g in gamma:
                # print('C = {}, gamma = {}'.format(c, g))
                acc = svm_train(y, x, '-q -s 0 -v 5 -t {} -c {} -g {}'.format(kernel_type[kernel], c, g))
                if(acc > max_acc):
                    max_acc = acc
                    best_param = (c, g)
    return best_param

def linearKernel(xi, xj):
    return np.dot( xi, xj.T)

# C = 10, gamma = 0.05
def RBFKernel(xi, xj, gamma):
    return np.exp(- gamma * cdist(xi, xj, metric='sqeuclidean'))

def newKernel(x1, x2, gamma=0.05):
    k = linearKernel(x1, x2) + RBFKernel(x1, x2, gamma)
    # in each row [number , feature1, feature2, ..., featureN ]
    return np.concatenate((np.arange(1, k.shape[0]+1)[:, np.newaxis], k), axis=1)

# C = 30, gamma = 0.2, degree = 2, coef0 = 0
def polynomialKernel(xi, xj, d, gamma, r):
    return (gamma * np.dot( xi, xj.T) + r) ** d

def poly_sigmoid_kernel(x1, x2, d, gamma, r):
    k = np.tanh(polynomialKernel(x1, x2, d, gamma, r))
    return np.concatenate((np.arange(1, k.shape[0]+1)[:, np.newaxis], k), axis=1)

def linear_sigmoid_kernel(x1, x2):
    k = np.tanh(linearKernel(x1, x2))
    return np.concatenate((np.arange(1, k.shape[0]+1)[:, np.newaxis], k), axis=1)

def rbf_sigmoid_kernel(x1, x2, gamma):
    k = np.tanh(RBFKernel(x1, x2, gamma))
    return np.concatenate((np.arange(1, k.shape[0]+1)[:, np.newaxis], k), axis=1)

############################### main ##############################
x_train = np.loadtxt('X_train.csv', dtype=float, delimiter=',')
y_train = np.loadtxt('Y_train.csv', dtype=int)
x_test = np.loadtxt('X_test.csv', dtype=float, delimiter=',')
y_test = np.loadtxt('Y_test.csv', dtype=int)

## part 1
print('Linear kernel:')
train_and_predict((x_train, x_test), (y_train, y_test), 'linear')
print('Polynomial kernel:')
train_and_predict((x_train, x_test), (y_train, y_test), 'polynomial')
print('RBF kernel:')
train_and_predict((x_train, x_test), (y_train, y_test), 'rbf')

## part 2
opt = ' -s 0 -c '
best_c = gridSearch(x_train, y_train, 'linear')
# # best_c = 10
print('Linear kernel parameter optimization:\n\tC = {}'.format(best_c))
train_and_predict((x_train, x_test), (y_train, y_test), 'linear', opt + '{}'.format(best_c))
print('---------------------------------------------------------------------')
best_c, best_g, best_d, best_r = gridSearch(x_train, y_train, 'polynomial')
# # (best_c, best_g, best_d, best_r) = (1, 0.05, 2, 0)
print('Polynomial kernel paramter optimization:\n\tC = {}, gamma = {}, degree = {}, coef0 = {}'.format(best_c, best_g, best_d, best_r))
train_and_predict((x_train, x_test), (y_train, y_test), 'polynomial', opt + '{} -g {} -d {} -r {}'.format(best_c, best_g, best_d, best_r))
print('---------------------------------------------------------------------')
best_c, best_g = gridSearch(x_train, y_train)
# # (best_c, best_g) = (10, 0.05)
print('RBF kernel parameter optimization:\n\tC = {}, gamma = {}'.format(best_c, best_g))
train_and_predict((x_train, x_test), (y_train, y_test), 'rbf', opt + '{} -g {}'.format(best_c, best_g))
print('---------------------------------------------------------------------')

## part 3
print('Linear kernel + Sigmoid kernel:')
train_kernel = newKernel(x_train, x_train)  # (5000, 5001)
test_kernel = newKernel(x_test, x_train)    # (2500, 5001)
train_and_predict((train_kernel, test_kernel), (y_train, y_test), 'self-defined')

# discussion
# print('Sigmoid kernel:')
# train_and_predict((x_train, x_test), (y_train, y_test), 'sigmoid')

# def my_gridSearch(x_train, x_test, y_train, y_test, kernel2comb):
#     kernel_type = {'linear':'0', 'polynomial':'1', 'rbf':'2', 'self-defined':'4'}
#     # C = [958, 960, 962, 964]
#     C = [1, 10, 20, 30]
#     gamma = [0.2, 0.1, 0.05, 0.01]
#     degree = [2, 3, 4, 5]
#     coef0 = [0, 5, 10]

#     max_acc = 0.0
#     opt = '-q -s 0 -v 5 -t 4 '
#     if(kernel2comb == 'linear'):
#         best_param = (C[0])
#         for c in C:
#             print('C = {}'.format(c))
#             train_kernel = linear_sigmoid_kernel(x_train, x_train)  # (5000, 5001)                
#             acc = svm_train(y_train, train_kernel, opt + '-c {}'.format(c))
#             if(acc > max_acc):
#                 max_acc = acc
#                 best_param = (c)
#         (c) = best_param
#         print('Sigmoid with Linear kernel parameter optimization:\n\tC = {}'.format(c))
#         train_kernel = linear_sigmoid_kernel(x_train, x_train)  # (5000, 5001)                
#         test_kernel = linear_sigmoid_kernel(x_test, x_train)    # (2500, 5001)
#         train_and_predict((train_kernel, test_kernel), (y_train, y_test), 'self-defined', ' -s 0 -c {}'.format(c))
#     elif(kernel2comb == 'polynomial'):
#         best_param = (C[0], gamma[0], coef0[0], degree[0])
#         for c in C:
#             for g in gamma:
#                 for d in degree:
#                     for r in coef0:
#                         print('C = {}, gamma = {}, degree = {}, coef0 = {}'.format(c, g, d, r))
#                         train_kernel = poly_sigmoid_kernel(x_train, x_train, d, g, r)  # (5000, 5001)                
#                         acc = svm_train(y_train, train_kernel, opt + '-c {} -g {} -d {} -r {}'.format( c, g, d, r))
#                         if(acc > max_acc):
#                             max_acc = acc
#                             best_param = (c, g, d, r)
#         (c, g, d, r) = best_param
#         print('Sigmoid with Polynomial kernel paramter optimization:\n\tC = {}, gamma = {}, degree = {}, coef0 = {}'.format(c, g, d, r))
#         train_kernel = poly_sigmoid_kernel(x_train, x_train, d, g, r)  # (5000, 5001)                
#         test_kernel = poly_sigmoid_kernel(x_test, x_train, d, g, r)    # (2500, 5001)
#         train_and_predict((train_kernel, test_kernel), (y_train, y_test), 'self-defined', ' -s 0 -c {} -g {} -d {} -r {}'.format( c, g, d, r))
#     elif(kernel2comb == 'rbf'):
#         best_param = (C[0], gamma[0])
#         for c in C:
#             for g in gamma:
#                 print('C = {}, gamma = {}'.format(c, g))
#                 train_kernel = rbf_sigmoid_kernel(x_train, x_train, g)  # (5000, 5001)                
#                 acc = svm_train(y_train, train_kernel, opt + '-c {} -g {}'.format( c, g))
#                 if(acc > max_acc):
#                     max_acc = acc
#                     best_param = (c, g)
#         (c, g) = best_param
#         print('Sigmoid with RBF kernel parameter optimization:\n\tC = {}, gamma = {}'.format(c, g))
#         train_kernel = rbf_sigmoid_kernel(x_train, x_train, g)  # (5000, 5001)                
#         test_kernel = rbf_sigmoid_kernel(x_test, x_train, g)    # (2500, 5001)
#         train_and_predict((train_kernel, test_kernel), (y_train, y_test), 'self-defined', ' -s 0 -c {} -g {}'.format( c, g))

# my_gridSearch(x_train, x_test, y_train, y_test, 'linear')
# c = 20
# print('Sigmoid with Linear kernel parameter optimization:\n\tC = {}'.format(c))
# train_kernel = linear_sigmoid_kernel(x_train, x_train)  # (5000, 5001)                
# test_kernel = linear_sigmoid_kernel(x_test, x_train)    # (2500, 5001)
# train_and_predict((train_kernel, test_kernel), (y_train, y_test), 'self-defined', ' -s 0 -c {}'.format(c))

# my_gridSearch(x_train, x_test, y_train, y_test, 'polynomial')
# (c, g, d, r) = (1, 0.01, 3, 0)
# print('Sigmoid with Polynomial kernel paramter optimization:\n\tC = {}, gamma = {}, degree = {}, coef0 = {}'.format(c, g, d, r))
# train_kernel = poly_sigmoid_kernel(x_train, x_train, d, g, r)  # (5000, 5001)                
# test_kernel = poly_sigmoid_kernel(x_test, x_train, d, g, r)    # (2500, 5001)
# train_and_predict((train_kernel, test_kernel), (y_train, y_test), 'self-defined', ' -s 0 -c {} -g {} -d {} -r {}'.format( c, g, d, r))

# my_gridSearch(x_train, x_test, y_train, y_test, 'rbf')
# (c, g) = (10, 0.05)
# print('Sigmoid with RBF kernel parameter optimization:\n\tC = {}, gamma = {}'.format(c, g))
# train_kernel = rbf_sigmoid_kernel(x_train, x_train, g)  # (5000, 5001)                
# test_kernel = rbf_sigmoid_kernel(x_test, x_train, g)    # (2500, 5001)
# train_and_predict((train_kernel, test_kernel), (y_train, y_test), 'self-defined', ' -s 0 -c {} -g {}'.format( c, g))