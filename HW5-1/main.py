import numpy as np
from numpy.linalg import inv, det
import matplotlib.pyplot as plt

def getKernel(x1, x2, theta):
    (sigma, alpha, length) = (theta[0], theta[1], theta[2])
    # print(theta[0], theta[1], theta[1])
    kernel = np.ones((len(x1), len(x2)), dtype=float)
    for n in range(len(x1)):
        for m in range(len(x2)):
            # rational quadratic kernels
            kernel[n][m] = (sigma**2) * ( 1 + (x1[n] - x2[m])**2 / (2.0 * alpha * (length**2)) )**(- alpha)
    return kernel

def getCovariance(x, theta, beta=5.0):
    return (getKernel(x, x, theta) + np.identity(len(x)) / beta)

def logLikelihood(theta, *args):
    (train_x, train_y) = args
    covar = getCovariance(train_x, theta)  # marginal likelihood
    # log(p(y|X)) = max( -0.5 * (y.T*C^(-1)*y + log(det(C)) + N*log(2*pi)))
    # log(p(y|X)) = min( 0.5 * (y.T*C^(-1)*y + log(det(C)) + N*log(2*pi)))
    return ( np.dot( np.dot(train_y.T, inv(covar)), train_y) +
            np.log(det(covar)) + len(train_x) * np.log(2 * np.pi) ) / 2.0

def predictDistribution(train_x, test_x, train_y, theta=np.ones(3), beta=5.0):
    covar = getCovariance(train_x, theta)  # marginal likelihood
    # k(x, x*)
    kernel = getKernel(train_x, test_x, theta)    
    # k* = k(x*, x*) + 1/beta
    k_test = getKernel(test_x, test_x, theta) + (1.0 / beta)

    ### predictive distribution (conditional distribution)
    # predict mean = k(x, x*).T * C^(-1) * y
    pred_mean = np.dot( np.dot(kernel.T, inv(covar)), train_y)
    # predict var = k* - k(x, x*).T * C^(-1) * k(x, x*)
    pred_var = k_test - np.dot( np.dot(kernel.T, inv(covar)), kernel)

    return pred_mean, pred_var

def plotResult(n_fig, mean, var, data):
    (train_x, train_y, test_x) = data    
    plt.figure(n_fig)
    std = np.diag(var) ** 0.5
    y1 = mean + 2 * std
    y2 = mean - 2 * std
    # plt.plot(test_x, y1, color='r')
    # plt.plot(test_x, y2, color='r')
    plt.fill_between(test_x, y1, y2, color='r', alpha=0.2)
    plt.scatter(train_x, train_y)
    plt.plot(test_x, mean, color='k')
    plt.xlim(-60, 60)

######################################### main ############################################
size = 34
train_x = np.empty(size, dtype=float)
train_y = np.empty(size, dtype=float)
f = open('input.data')
for i, l in enumerate(f):
    train_x[i], train_y[i] = l.strip('\n').split()

# generate test data
test_x = np.linspace(-60, 60, num=500, dtype=float)
theta_0 = np.ones(3, dtype=float)
pred_mean, pred_var = predictDistribution(train_x, test_x, train_y, theta_0)
plotResult(1, pred_mean, pred_var, (train_x, train_y, test_x))
plt.title('Gaussian Process Regression w/o Optimization\n(σ: {}, α: {}, ℓ: {})'.format(theta_0[0], theta_0[1], theta_0[2]))
print('Gaussian Process Regression without Optimization (σ: {}, α: {}, ℓ: {})'.format(theta_0[0], theta_0[1], theta_0[2]))
# plt.ylim(-4.5, 4)

from scipy.optimize import minimize
opt_param = minimize(logLikelihood, args=(train_x, train_y), x0=np.ones(3, dtype=float), method='CG').x
opt_mean, opt_var = predictDistribution(train_x, test_x, train_y, theta=opt_param)
plotResult(2, opt_mean, opt_var, (train_x, train_y, test_x))
plt.title('Gaussian Process Regression with Optimization\n(σ: {:.1f}, α: {:.1f}, ℓ: {:.1f})'.format(opt_param[0], opt_param[1], opt_param[2]))
# plt.title('Gaussian Process Regression with Optimization\n(σ: {}, α: {}, ℓ: {})'.format(opt_param[0], opt_param[1], opt_param[2]))
print('Gaussian Process Regression without Optimization (σ: {}, α: {}, ℓ: {})'.format(opt_param[0], opt_param[1], opt_param[2]))
# plt.ylim(-4.5, 4)


# tmp = opt_param.copy()
# tmp[0] = 1.0
# pred_mean, pred_var = predictDistribution(train_x, test_x, train_y, tmp)
# plotResult(3, pred_mean, pred_var, (train_x, train_y, test_x))
# plt.title('Gaussian Process Regression w/o Optimization\n(σ: {}, α: {}, ℓ: {})'.format(tmp[0], tmp[1], tmp[2]))
# print('Gaussian Process Regression without Optimization (σ: {}, α: {}, ℓ: {})'.format(tmp[0], tmp[1], tmp[2]))
# plt.ylim(-4.5, 4)

# tmp2 = opt_param.copy()
# tmp2[1] = 1.0
# pred_mean, pred_var = predictDistribution(train_x, test_x, train_y, tmp2)
# plotResult(4, pred_mean, pred_var, (train_x, train_y, test_x))
# plt.title('Gaussian Process Regression w/o Optimization\n(σ: {}, α: {}, ℓ: {})'.format(tmp2[0], tmp2[1], tmp2[2]))
# print('Gaussian Process Regression without Optimization (σ: {}, α: {}, ℓ: {})'.format(tmp2[0], tmp2[1], tmp2[2]))
# plt.ylim(-4.5, 4)


# tmp3 = opt_param.copy()
# tmp3[2] = 1.0
# pred_mean, pred_var = predictDistribution(train_x, test_x, train_y, tmp3)
# plotResult(5, pred_mean, pred_var, (train_x, train_y, test_x))
# plt.title('Gaussian Process Regression w/o Optimization\n(σ: {}, α: {}, ℓ: {})'.format(tmp3[0], tmp3[1], tmp3[2]))
# print('Gaussian Process Regression without Optimization (σ: {}, α: {}, ℓ: {})'.format(tmp3[0], tmp3[1], tmp3[2]))
# plt.ylim(-4.5, 4)


plt.show()