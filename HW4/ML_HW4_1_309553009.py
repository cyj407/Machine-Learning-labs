import numpy as np
from numpy.linalg import det, inv
import argparse
import math
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

def GaussianDataGen(mean, var):
    # box-muller
    u = np.random.uniform(0.0, 1.0)
    v = np.random.uniform(0.0, 1.0)
    z = np.sqrt(-2.0 * np.log(u)) * np.sin(2 * np.pi * v)
    return mean + np.sqrt(var) * z


def sigmoid(x):
    for i in range(x.shape[0]):
        try:
            x[i] = 1.0 / (1.0 + math.exp(-1.0 * x[i]))
        except OverflowError:
            x[i] = 0.0
    return x


def converge(new_w, w):
    for i in range(len(w)):
        if(abs(w[i] - new_w[i]) > 5e-4):
            return False
    return True


def ConfusionMat(y, pred_y):
    tp = fp = fn = tn = 0
    for _y, _py in zip(y, pred_y):
        if(_y == 1.0 and _py == 1.0):
            tp += 1
        elif(_y == 1.0 and _py == 0.0):
            fn += 1            
        elif(_y == 0.0 and _py == 1.0):
            fp += 1
        elif(_y == 0.0 and _py == 0.0):
            tn += 1
    return tp, fp, fn, tn


def printRes(X, y, w):
    pred_val = sigmoid(np.dot(X, w))
    pred_y = np.where(pred_val > 0.5, 1.0, 0.0)
    # print(pred_y)

    tp, fp, fn, tn = ConfusionMat(y, pred_y)
    sensitivity = float(tp / (tp+fn))
    specificity = float(tn / (tn+fp))

    print('w:')
    for i in range(len(w)):
        print('{:.10f}'.format(w[i]))
    print('\nConfusion Matrix:')
    print("              Predict cluster 1 Predict cluster 2")
    print("Is cluster 1         {}                {}".format(tp, fn))
    print("Is cluster 2         {}                {}\n".format(fp, tn))
    print('Sensitivity (Successfully predict cluster 1): {:.5f}'.format(sensitivity))
    print('Specificity (Successfully predict cluster 2): {:.5f}\n'.format(specificity))
    return pred_y


def GradientDescent(X, y, w):
    lr_rate = 1
    for cnt in range(50000):
        # w_n+1 = w_n + gradient(X^T * (y - sigmoid(x*w)))
        g = np.dot( X.T, (y - sigmoid(np.dot(X, w))))
        new_w = w + lr_rate * g
        if(converge(new_w, w)):
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!converge')
            break
        w = new_w
    
    print('Gradient descent:\n')
    pred_y = printRes(X, y, new_w)
    return pred_y


def NewtonMethod(X, y, w):
    lr_rate = 1
    for cnt in range(50000):
        xw = np.dot(X, w)
        D = np.eye( X.shape[0], dtype=float)
        for i in range(D.shape[0]):
            try:
                D[i][i] = math.exp(-1.0 * xw[i]) / (1.0 + math.exp(-1.0 * xw[i])) **2
            except OverflowError:
                D[i][i] = 0.0

        hessian = np.dot( np.dot(X.T, D), X)    # X^T * D * X
        g = np.dot( X.T, (y - sigmoid(xw)))
        
        if(det(hessian) != 0):
            # x1 = x0 + Hessian^(-1) * gradient
            new_w = w + np.dot( inv(hessian), g)
        else:
            ## use Steepest gradient descent singular when not invertible --> determinant == 0
            new_w = w + lr_rate * g

        if(converge(new_w, w)):
            # print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!converge')
            break   
        w = new_w
    
    print('Newton\'s method:\n')
    pred_y = printRes(X, y, new_w)

    return pred_y


def getPredData(X, pred):
    c1 = []
    c2 = []
    for x, p in zip(X, pred):
        if(p == 0.0):
           c1.append((x[0], x[1]))
        else: 
           c2.append((x[0], x[1])) 
    return c1, c2 


def LogisticRegression():
    D1 = [(GaussianDataGen(args.mx1, args.vx1), GaussianDataGen(args.my1, args.vy1)) for i in range(args.N)]
    D2 = [(GaussianDataGen(args.mx2, args.vx2), GaussianDataGen(args.my2, args.vy2)) for i in range(args.N)]
    X = np.array([[dx, dy, 1.0] for (dx, dy) in D1] + [[dx, dy, 1.0] for (dx, dy) in D2])   # (100, 3)
    y = np.array([0.0] * args.N + [1.0] * args.N)   # (100,)

    init_w = np.zeros(3, dtype=float)
    pred_g = GradientDescent(X, y, init_w)

    print('----------------------------------------------------------')

    pred_n = NewtonMethod(X, y, init_w)

    # visualize
    import matplotlib.pyplot as plt
    plt.subplot(1, 3, 1)
    plt.title('Ground truth')
    plt.scatter(*zip(*D1), color='r')
    plt.scatter(*zip(*D2), color='b')

    plt.subplot(1, 3, 2)
    plt.title('Gradient descent')
    g1, g2 = getPredData(X, pred_g)
    plt.scatter(*zip(*g1), color='r')
    plt.scatter(*zip(*g2), color='b')    

    plt.subplot(1, 3, 3)
    plt.title('Newton\'s method')    
    n1, n2 = getPredData(X, pred_n)
    plt.scatter(*zip(*n1), color='r')
    plt.scatter(*zip(*n2), color='b')    

    plt.show()

################################# main ###################################

parser = argparse.ArgumentParser()
parser.add_argument('--N', default=50)
parser.add_argument('--mx1', default=1)
parser.add_argument('--vx1', default=2)
parser.add_argument('--my1', default=1)
parser.add_argument('--vy1', default=2)
parser.add_argument('--mx2', default=3) # 10 / 3
parser.add_argument('--vx2', default=4) # 2 / 4
parser.add_argument('--my2', default=3) # 10 / 3
parser.add_argument('--vy2', default=4) # 2 / 4
args = parser.parse_args()

LogisticRegression()
