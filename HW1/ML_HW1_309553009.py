import numpy as np
import matplotlib.pyplot as plt

## initialize the matrix with all zeros
def initMat(n, init_val=0.0):
    M = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(init_val)
        M.append(row)
    return np.array(M)

## return a identity matrix I
def identityMat(n):
    I = []
    for i in range(n):
        row = []
        for j in range(n):
            if(i == j):
                row.append(1.0)
            else:
                row.append(0.0)
        I.append(row)
    return (np.array(I))


## compute transpose matrix M
def transMat(M):
    trans_M = []
    for col in range(M.shape[1]):
        trans_row = []
        for row in range(M.shape[0]):
            trans_row.append(M[row][col])
        trans_M.append(trans_row)
    return np.array(trans_M)


## compute inverse matrix L
def inverseMatL(L):
    inv_L = initMat(len(L))
    for i in range(len(L)):
        inv_L[i][i] = float(1.0/L[i][i])
        for j in range(i):
            s = 0
            for k in range(j, i):
                s = s + L[i][k] * inv_L[k][j]
            inv_L[i][j] = float(-s / L[i][i])
    # print(inv_L)
    return inv_L


def inverseMatU(U):
    inv_U = initMat(len(U))
    for i in range(len(U)-1, -1, -1):
        inv_U[i][i] = float(1.0/U[i][i])
        for j in range(len(U)-1, i, -1):
            s = 0
            for k in range( i+1, j+1):
                s = s + U[i][k] * inv_U[k][j]
            inv_U[i][j] = float(-s / U[i][i])
    # print(inv_U)
    return inv_U

def LU_inverse(A, base_n): # use LU decomposition
    ## initialize L and U
    L = initMat(base_n)
    U = initMat(base_n)

    for i in range(base_n):
        U[0][i] = A[0][i]
        L[i][i] = 1.0   # L diagonal = 1.0
        # print('A{}{} = U{}{}'.format(0, i, 0, i))

    for i in range(1, base_n):
        L[i][0] = float(A[i][0]/U[0][0])
        # print('A{}{} = L{}{}*U{}{}'.format(i, 0, i, 0, 0, 0))

    for i in range(1, base_n):  # control row of U, and column of L

        for j in range(1, base_n):
            tmp = 0.0
            ss = ' '
            for k in range(i):
                tmp = tmp + L[i][k] * U[k][j]
                ss = ss + 'L{}{}*U{}{}+'.format(i,k,k,j)
            U[i][j] = A[i][j] - tmp
            # print('A{}{} = '.format(i, j) + ss + 'U{}{}'.format(i, j))

        for j in range( i+1, base_n):
            tmp = 0.0
            ss = ' '
            for k in range(i):
                tmp = tmp + L[j][k] * U[k][i]
                ss = ss + 'L{}{}*U{}{}+'.format(j,k,k,i)            
            L[j][i] = float((A[j][i] - tmp) / U[i][i])
            # print('A{}{} = '.format(j, i) + ss + 'L{}{}*U{}{}'.format( j, i, i, i))

    # print(np.dot(L, U)) # L == U

    ## A = LU --> inverse(A) = inverse(LU) = inverse(U)*inverse(L)
    return np.dot(inverseMatU(U), inverseMatL(L))


def LSE(A, b, base_n, lse_lambda):

    # LU decomposition --> find inverse(A^T A + lambda)
    inv_design = LU_inverse(np.dot(transMat(A), A) + lse_lambda * identityMat(base_n), base_n)
    # print(inv_design)

    ## X = inverse(A^T * A + lambda * I) * A^T * b = inv_design * A^T * b
    X = np.dot( np.dot(inv_design, transMat(A)), b)
    
    # print and plot
    printRes('LSE:', X, A, b)
    return X


def Newton(A, b, base_n):

    xn = np.array([0.0 for i in range(base_n)])

    # delta_f(X_n) = 2 * A^T * A * X_n - 2 * A^T * b
    delta_f = 2 * np.dot( np.dot(transMat(A), A), xn) - 2 * np.dot(transMat(A), b)
    
    # Hf^(-1) = inverse(2 * A^T * A)
    inv_Hf = LU_inverse(2 * np.dot(transMat(A), A), base_n)

    # X_n+1 = X_n - delta_f(X_n) * Hf^(-1)
    X = xn - np.dot(delta_f, inv_Hf)

    # print and plot
    printRes('Newton\'s Method:', X, A, b)
    return X

def getError(A, X, b):
    ## ||AX - b||^2 = (AX-b)(AX-b)^T
    tmp = np.dot(A, X) - b
    tmp = tmp.reshape(len(tmp), -1)
    # print(tmp.shape)
    return np.dot(transMat(tmp), tmp)[0][0]

def printRes(m_name, X, A, b):
    print(m_name)
    # print equations
    equ = ""
    for i in range(len(X)):
        if(i != 0 and X[i] > 0):
            equ = equ + '+ '
        if(i != len(X)-1):
            equ = equ + '{}X^{} '.format(X[i], len(X)-i-1)
        else:
            equ = equ + '{} '.format(X[i])
    print('Fitting line: ' + equ)
    print('Total error: {}\n'.format(getError(A, X, b)))

def getFitY(s_x, X):
    s_y = 0
    for i in range(len(X)):
        s_y = s_y + X[i] * (s_x ** (len(X)-i-1))
    return s_y

##################################### main ##########################################

## input file
file_n = str((input("file path: ") or 'testfile.txt'))
pts_list = []
with open(file_n) as f:
    for l in f:
        pt = l.strip().split(',')
        pts_list.append(pt)

# base_n = 3, highest degree of the polynomial is 2
base_n = int((input("n = ") or 3))
lse_lambda = int((input("lambda = ") or 0))


## compute A
A = []
for pt in pts_list:
    x = float(pt[0])
    row = []
    for deg in range( base_n-1, -1, -1):
        row.append(x ** deg)
    A.append(row)

A = np.array(A)
b = np.array([float(pt[1]) for pt in pts_list])
# print(b)

X_LSE = LSE(A, b, base_n, lse_lambda)
X_Newton = Newton(A, b, base_n)

## visualize
x = [float(xi) for xi,yi in pts_list]
y = [float(yi) for yi,yi in pts_list]

plt.figure(1)
plt.title('LSE')
plt.scatter(x, y, color='r')
fit_x = np.linspace(int(min(x))-2, int(max(x))+2, len(x)*5)
fit_y = getFitY(fit_x, X_LSE)
plt.plot(fit_x, fit_y, color='k')
plt.xlim(int(min(x))-2, int(max(x)+2))

plt.figure(2)
plt.title('Newton\'s Method')
plt.scatter(x, y, color='r')
fit_x = np.linspace(int(min(x))-2, int(max(x))+2, len(x)*5)
fit_y = getFitY(fit_x, X_Newton)
plt.plot(fit_x, fit_y, color='k')

plt.xlim(int(min(x))-2, int(max(x)+2))
plt.show()
