import numpy as np
import os
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

W = 195
H = 231
SHAPE = ( int(W/5), int(H/5))


def loadDataset(dir_name):
    f_list = os.listdir(dir_name)
    img_list = []
    for f in f_list:
        path = dir_name + f
        img = Image.open(path)
        (w, h) = img.size
        img = img.resize((int(w/5), int(h/5)), Image.BILINEAR)
        img = np.asarray(img).reshape(-1)
        img_list.append(img)
    return np.array(img_list).astype(float), f_list

def plotFaces(folder, name, W): 
    plt.clf()
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            plt.subplot(5, 5, idx + 1)
            plt.imshow( W[:, idx].reshape(SHAPE[::-1]), cmap='gray')
            plt.axis('off')
    path = './{}/{}.png'.format(folder, name)    
    plt.savefig(path)


def plotReconstruct(folder, name, recon):
    plt.clf()
    for i in range(2):
        for j in range(5):
            idx = i * 5 + j
            plt.subplot(2, 5, idx + 1)
            plt.imshow( recon[idx].reshape(SHAPE[::-1]), cmap='gray')
            plt.axis('off')
    path = './{}/{}.png'.format(folder, name)    
    plt.savefig(path)


def PCA(X, k, task, labels=None):

    mean = np.mean(X, axis=0)
    diff = (X - mean)
    S = np.dot( diff, diff.T)      # S: covariance matrix
    
    eig_val, eig_vec = np.linalg.eigh(S)     # S is symmetric
    eig_vec = np.dot( diff.T, eig_vec)

    norm = np.linalg.norm(eig_vec, axis=0)
    eig_vec /= norm

    sort_eig_val = np.argsort(eig_val)[::-1]        # ascending --> descending
    eig_vec = eig_vec[:, sort_eig_val]      # reorder


    W = eig_vec[:, :k].real         # (1794, 25)


    ## part1
    if(task == 1):
        folder = 'PCA_{}'.format(datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.mkdir(folder)
        
        plotFaces(folder, 'eigenface', W)

        r = np.random.choice( X.shape[0], size=10)
        original = X[r]
        
        # plot original
        print(labels[r])
        for idx, name in enumerate(labels[r]):
            path = './Yale_Face_Database/Training/' + name
            if(not os.path.isfile(path)):
                path = './Yale_Face_Database/Testing/' + name
            img = Image.open(path)
            (w, h) = img.size
            # img = img.resize((int(w/5), int(h/5)), Image.BILINEAR)
            img = np.asarray(img).astype(float)
            plt.subplot(2, 5, idx + 1)
            plt.imshow( img, cmap='gray')
            plt.axis('off')
        path = './{}/original.png'.format(folder)    
        plt.savefig(path)

        reconstruct = np.dot( np.dot( (original - mean), W), W.T) + mean

        plotReconstruct(folder, 'reconstruct', reconstruct)

    return W, mean

def getClass(_list):
    c = np.array([int(name[7:9]) for name in _list]).astype(int)
    return c

def LDA(X, y, k, task):
    feature_num = X.shape[1]       # features: pixels
    mean = np.mean(X, axis=0)

    # class mean
    classes, cls_cnt = np.unique(y, return_counts=True)   # show all class
    cls_mean = np.zeros((len(classes), feature_num))
    for i, c in enumerate(classes):
        cls_mean[i] = np.mean(X[y == c], axis=0)

    # compute sw
    sw = np.zeros((feature_num, feature_num))
    for i, c in enumerate(classes):
        x_to_mean = X[y == c] - cls_mean[i]
        sw += np.dot( x_to_mean.T, x_to_mean)
    
    # compute sb
    sb = np.zeros((feature_num, feature_num), dtype=float)
    for i, c in enumerate(classes):
        cls_to_mean = cls_mean[i] - mean
        sb += cls_cnt[i] * np.dot( cls_to_mean.T, cls_to_mean)

    # eigen
    S = np.dot( np.linalg.pinv(sw), sb)
    eig_val, eig_vec = np.linalg.eig(S)     # S is symmetric

    norm = np.linalg.norm(eig_vec, axis=0)
    eig_vec /= norm

    sort_eig_val = np.argsort(eig_val)[::-1]        # ascending --> decending
    eig_vec = eig_vec[:, sort_eig_val]      # reorder

    W = eig_vec[:, :k].real         # (1794, 25)

    if(task == 1):
        folder = 'LDA_{}'.format(datetime.now().strftime('%Y%m%d-%H%M%S'))
        os.mkdir(folder)
        
        plotFaces(folder, 'fisherface', W)
        
        r = np.random.choice( X.shape[0], size=10)
        original = X[r]
        reconstruct = np.dot( np.dot( (original), W), W.T)
        
        plotReconstruct(folder, 'reconstruction', reconstruct)
    
    return W

def kNN(train_x, train_y, test_x, test_y, k):
    dist = cdist(test_x, train_x, metric='sqeuclidean')
    near_to_far = np.argsort(dist, axis=1)  # (30, 135)

    correct = 0
    pred = np.zeros_like(test_y)
    for i in range(near_to_far.shape[0]):
        pred_k = train_y[near_to_far[i]][:k]
        cls_list, cls_cnt = np.unique(pred_k, return_counts=True)
        pred = cls_list[np.argmax(cls_cnt)]
    
        if(pred == test_y[i]):
            correct += 1
    print('K={}, accuracy: {:3f} ({}/{})'.format(k, float(correct/len(test_y)), correct, len(test_y)))

def linearKernel(X):
    return np.dot( X, X.T)

def polynomialKernel(X, gamma, deg):
    return np.power( (gamma * np.dot(X, X.T)), deg)

def RBFKernel(X, gamma):
    return np.exp(-1.0 * gamma * cdist(X, X, metric='sqeuclidean'))

def getKernel(kernel, X, param):
    if(kernel == 'linear'):
        return linearKernel(X)
    if(kernel == 'polynomial'):
        (gamma, deg) = param
        return polynomialKernel(X, gamma, deg)
    if(kernel == 'rbf'):
        (gamma) = param
        return RBFKernel(X, gamma)

def kernelPCA(X, k, kernel_type, kernel_param=None):
    
    kernel = getKernel(kernel_type, X, kernel_param)
    
    # centralize
    # l = np.ones_like(kernel) * (1.0 / kernel.shape[0])
    # kernel = kernel - np.dot( l, kernel) - np.dot( kernel, l) + np.dot( np.dot( l, kernel), l)

    eig_val, eig_vec = np.linalg.eigh(kernel)     # S is symmetric
    norm = np.linalg.norm(eig_vec, axis=0)
    eig_vec /= norm

    sort_eig_val = np.argsort(eig_val)[::-1]        # ascending --> decending
    eig_vec = eig_vec[:, sort_eig_val]      # reorder

    W = eig_vec[:, :k].real         # (1794, 25)
    W = np.dot( kernel, W)
    return W


def kernelLDA(X, y, k, kernel_type, kernel_param=None):
    kernel = getKernel(kernel_type, X, kernel_param)

    feature_num = kernel.shape[1]       # features: kernel
    mean = np.mean(kernel, axis=0)

    # class mean
    classes, cls_cnt = np.unique(y, return_counts=True)   # show all class
    cls_mean = np.zeros((len(classes), feature_num))
    for i, c in enumerate(classes):
        cls_mean[i] = np.mean(kernel[y == c], axis=0)

    # compute sw
    sw = np.zeros((feature_num, feature_num))
    for i, c in enumerate(classes):
        kernel_to_mean = kernel[y == c] - cls_mean[i]
        sw += np.dot( kernel_to_mean.T, kernel_to_mean)
    
    # compute sb
    sb = np.zeros((feature_num, feature_num), dtype=float)
    for i, c in enumerate(classes):
        cls_to_mean = cls_mean[i] - mean
        sb += cls_cnt[i] * np.dot( cls_to_mean.T, cls_to_mean)

    # eigen
    S = np.dot( np.linalg.pinv(sw), sb)
    eig_val, eig_vec = np.linalg.eig(S)     # S is symmetric

    norm = np.linalg.norm(eig_vec, axis=0)
    eig_vec /= norm

    sort_eig_val = np.argsort(eig_val)[::-1]        # ascending --> decending
    eig_vec = eig_vec[:, sort_eig_val]      # reorder

    W = eig_vec[:, :k].real         # (1794, 25)
    W = np.dot( kernel, W)
    return W

###################################### main ###################################
## PCA, LDA
train_set, train_name = loadDataset('./Yale_Face_Database/Training/')   # (135, 231, 195)
test_set, test_name = loadDataset('./Yale_Face_Database/Testing/')     # (30, 231, 195)
X = np.concatenate((train_set, test_set), axis=0)    # train+test
y = getClass(train_name+test_name)
train_y = getClass(train_name)
test_y = getClass(test_name)

# 1
PCA(X, k=25, task=1, labels=np.array(train_name+test_name))
LDA(X, y, k=25, task=1)
train_y = getClass(train_name)
PCA(train_set, k=25, task=1)
LDA(train_set, train_y, k=25, task=1)

'''---------------------------------------------------------------------------'''
# 2
print('PCA')
W_PCA, mean = PCA(X, k=25, task=2)
train_w_pca = np.dot( (train_set - mean), W_PCA)
test_w_pca = np.dot( (test_set - mean), W_PCA)
for i in range(1, 11):
    kNN(train_w_pca, train_y, test_w_pca, test_y, k=i)  # 0.9

print('LDA')
W_LDA = LDA(X, y, k=25, task=2)
train_w_lda = np.dot( train_set, W_LDA)
test_w_lda = np.dot( test_set, W_LDA)
for i in range(1, 11):
    kNN(train_w_lda, train_y, test_w_lda, test_y, k=i)  # 0.9
# '''---------------------------------------------------------------------------'''
# # 3
# W_kPCA = kernelPCA(X, k=25, kernel_type='rbf', kernel_param=(1e-7))
# W_kPCA = kernelPCA(X, k=25, kernel_type='polynomial', kernel_param=(1, 2))
print('linear kernel PCA')
W_kPCA = kernelPCA(X, k=100, kernel_type='linear')
train_kPCA = W_kPCA[:train_set.shape[0]]
test_kPCA = W_kPCA[train_set.shape[0]:]
for i in range(1, 11):
    kNN(train_kPCA, train_y, test_kPCA, test_y, k=i)  # 0.9

print()
print('polynomial kernel PCA')
W_kPCA = kernelPCA(X, k=100, kernel_type='polynomial', kernel_param=(1, 2))
train_kPCA = W_kPCA[:train_set.shape[0]]
test_kPCA = W_kPCA[train_set.shape[0]:]
for i in range(1, 11):
    kNN(train_kPCA, train_y, test_kPCA, test_y, k=i)  # 0.9

print()
print('RBF kernel PCA')
W_kPCA = kernelPCA(X, k=100, kernel_type='rbf', kernel_param=(1e-7))
train_kPCA = W_kPCA[:train_set.shape[0]]
test_kPCA = W_kPCA[train_set.shape[0]:]
for i in range(1, 11):
    kNN(train_kPCA, train_y, test_kPCA, test_y, k=i)  # 0.9

# W_kLDA = kernelLDA(X, y, k=25, kernel_type='rbf', kernel_param=(1e-7))
# W_kLDA = kernelLDA(X, y, k=25, kernel_type='polynomial', kernel_param=(1, 10))

print()
print('linear kernel LDA')
W_kLDA = kernelLDA(X, y, k=100, kernel_type='linear')
train_kLDA = W_kLDA[:train_set.shape[0],:]
test_kLDA = W_kLDA[train_set.shape[0]:,:]
for i in range(1, 11):
    kNN(train_kLDA, train_y, test_kLDA, test_y, k=i)  # 0.9


print()
print('polynomial kernel LDA')
W_kLDA = kernelLDA(X, y, k=100, kernel_type='polynomial', kernel_param=(0.1, 1))
train_kLDA = W_kLDA[:train_set.shape[0],:]
test_kLDA = W_kLDA[train_set.shape[0]:,:]
for i in range(1, 11):
    kNN(train_kLDA, train_y, test_kLDA, test_y, k=i)  # 0.9

print()
print('RBF kernel LDA')
W_kLDA = kernelLDA(X, y, k=100, kernel_type='rbf', kernel_param=(1e-7))
train_kLDA = W_kLDA[:train_set.shape[0],:]
test_kLDA = W_kLDA[train_set.shape[0]:,:]
for i in range(1, 11):
    kNN(train_kLDA, train_y, test_kLDA, test_y, k=i)  # 0.9