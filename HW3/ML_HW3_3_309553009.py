import numpy as np
from numpy.linalg import inv

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

def BaysianLinearRegression(b, n, a, w):
    cnt = 1
    m = np.zeros([n, 1])        # inital prior mean (nx1)
    S = b * np.identity(n)      # inital inverse of prior variance
    pred_mean = 0.0
    pred_var = 0.0

    X_list = []
    Y_list = []
    while(1):
        pt_x, y = PolynomialDataGen(n, a, w)
        X_list.append(pt_x)
        Y_list.append(y)

        X = np.array([(pt_x ** deg)  for deg in range(n)]).reshape(1, -1)

        # posterior Λ = 1/a * X^T * X + (bI or S)
        _lambda = float(1.0 / a) * np.dot( X.T, X) + S
        post_var = inv(_lambda)

        # posterior μ = Λ^(-1)*(1/a * X^T * y + S * m)  # 1st iteration: S * m == 0
        post_mean = np.dot( inv(_lambda), float(1.0/a) * X.T * y + np.dot(S, m))

        old_pred_mean = pred_mean
        old_pred_var = pred_var

        # predictive distribution ~ N(X*μ, a+X*Λ^(-1)*X^T)
        pred_mean = np.dot(X, post_mean)
        pred_var = a + np.dot( X, np.dot(inv(_lambda), X.T))

        print('Add data point ({:.5f}, {:.5f}):\n'.format(pt_x, y))
        print('Posterior mean:')     # n x 1
        for i in range(n):
            print('  {:.10f}'.format(post_mean[i][0]))
        print('\nPosterior variance:')    # n x n
        for i in range(n):
            for j in range(n):
                print('  {:.10f}'.format(post_var[i][j]), end=',')
            print()
        print('\nPredictive distribution ~ N({:.5F}, {:.5F})'.format(pred_mean[0][0], pred_var[0][0]))
        print('-----------------------------------------------')
        
        if(cnt == 10):
            m_10, S_10 = post_mean, post_var
        elif(cnt == 50):
            m_50, S_50 = post_mean, post_var

        if(cnt >= 100 and abs(pred_var - old_pred_var) < 1e-4 and abs(pred_mean - old_pred_mean) < 1e-2):
            m_final, S_final = post_mean, post_var
            break

        S = _lambda    # inverse of prior covariance 
        m = post_mean   # prior mean
        cnt += 1
    
    # visualize
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, sharex=True)
    fig.tight_layout(pad=3.0)
    sample_x = np.arange(-2.0, 2.0, 0.00001)
    
    plt.subplot(2, 2, 1)
    plt.title('Ground truth')
    sample_y = GroundTruth(sample_x, w, n)
    plt.plot(sample_x, sample_y, color='k')
    plt.plot(sample_x, sample_y + a, color='r')
    plt.plot(sample_x, sample_y - a, color='r')
    plt.xlim(-2, 2)
    plt.ylim(-20, 25)

    plt.subplot(2, 2, 2)
    plt.title('Predict result')
    sample_y_final, err_final = PredictResult(sample_x, m_final, S_final, n, a)
    plt.plot(sample_x, sample_y_final, color='k')
    plt.plot(sample_x, sample_y_final + err_final, color='r')
    plt.plot(sample_x, sample_y_final - err_final, color='r')
    plt.scatter(X_list, Y_list, color='tab:blue', s=6)
    plt.xlim(-2, 2)
    plt.ylim(-20, 25)

    plt.subplot(2, 2, 3)
    plt.title('After 10 incomes')
    sample_y_10, err_10 = PredictResult(sample_x, m_10, S_10, n, a)
    plt.plot(sample_x, sample_y_10, color='k')
    plt.plot(sample_x, sample_y_10 + err_10, color='r')
    plt.plot(sample_x, sample_y_10 - err_10, color='r')
    plt.scatter(X_list[:10], Y_list[:10], color='tab:blue',s=6)
    plt.xlim(-2, 2)
    plt.ylim(-20, 25)

    plt.subplot(2, 2, 4)
    plt.title('After 50 incomes')
    sample_y_50, err_50 = PredictResult(sample_x, m_50, S_50, n, a)
    plt.plot(sample_x, sample_y_50, color='k')
    plt.plot(sample_x, sample_y_50 + err_50, color='r')
    plt.plot(sample_x, sample_y_50 - err_50, color='r')
    plt.scatter(X_list[:50], Y_list[:50], color='tab:blue', s=6)
    plt.xlim(-2, 2)
    plt.ylim(-20, 25)

    plt.show()


def GroundTruth(x, w, n):
    y = np.zeros(len(x))
    for i in range(len(x)):
        deg_x = np.array([x[i] ** deg for deg in range(n)]).reshape(-1)
        w = np.array(w).reshape(-1)
        y[i] = np.dot( w.T, deg_x)
    return y

def PredictResult(x, m, s, n, a):
    y = np.zeros(len(x))
    err = np.zeros(len(x))
    for i in range(len(x)):
        X = np.array([x[i] ** deg for deg in range(n)])        
        # predictive distribution ~ N(X*μ, a+X*Λ^(-1)*X^T)
        y[i] = np.dot(X, m)
        err[i] = a + np.dot( X, np.dot( s, X.T))
    return y, err

################################ main ###################################
### 3
b = float(input('b = '))    # precision
n = int(input('n = '))
a = float(input('a = '))
print('w ({}x1 vector) :'.format(n))
w = [float(input('  w{} = '.format(i))) for i in range(n)]

BaysianLinearRegression(b, n, a, w)