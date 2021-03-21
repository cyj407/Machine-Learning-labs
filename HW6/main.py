import numpy as np
from cv2 import imread  # read image
from scipy.spatial.distance import cdist
from numpy.linalg import eig, inv
import matplotlib.pyplot as plt
import os
from PIL import Image

def kernel(s, c, g_s=2.5, g_c=2.5):
    s_rbf = cdist(s, s, 'sqeuclidean')
    c_rbf = cdist(c, c, 'sqeuclidean')
    return np.exp(-g_s * s_rbf) * np.exp(-g_c * c_rbf)

def kernelKMeans(data, k=2, init='spatial_nearest'):
    alpha_list = []
    n = data.shape[0] * data.shape[1]   # 10000
    flat_data = data.reshape(-1, 3) # (10000, 3)
    flat_data = flat_data / 255.0

    # spatial coordinates   # (10000, 2)
    indices = np.array([[i, j] for j in range(data.shape[1]) for i in range(data.shape[0])])
    indices = indices / 99.0

    # initialize, gram matrix and alpha
    gram_mat = kernel(indices, flat_data)   # (10000, 10000)

    if(init == 'random'):   # random assign classes
        alpha = np.random.randint(k, size=n)
    elif(init == 'spatial_nearest'):        
        # initialize cluster
        mean_idx = np.random.choice(n, size=k)
        mean_indices = indices[mean_idx]
        # classify by the nearest center
        dis = cdist( indices, mean_indices, 'sqeuclidean')
        # alpha[n][k] = n^th point belongs to k^th class
        alpha = np.argmin(dis, axis=1)  # k classes : class 0 ~ k-1     # (10000, 2)
    elif(init == 'kmeans++'):
        mean_idx = np.random.choice(n, size=k)
        means = flat_data[mean_idx]
        for i in range(1, k):
            dis = cdist(flat_data, means, 'sqeuclidean')
            min_dis = np.min(dis, axis=1)
            # compute the probability that each point occurs
            next_mean_idx = np.random.choice(n, size=1, p=min_dis/np.sum(min_dis))
            means[i] = flat_data[next_mean_idx]    
        # cluster
        dis = cdist(flat_data, means, 'sqeuclidean')
        alpha = np.argmin(dis, axis=1)  # k classes : class 0 ~ k-1     # (10000, 2)
    
    alpha_list.append(alpha.reshape(100, 100))

    for iteration in range(1, 51):
        # k(xj, xj)
        _first = np.diag(gram_mat).reshape(-1, 1)

        # - 2/|Ck| * sum(alpha_kn * k(xj, xn))
        num_C = np.zeros(k, dtype=float)
        for i in range(k):
            num_C[i] = np.count_nonzero(alpha == i)
        _second = np.zeros((n, k), dtype=float)
        for c in range(k):
            _second[:, c] = np.sum(gram_mat[:, alpha == c], axis=1)
        _second *= (-2.0 / num_C)

        # 1.0 / (k**2) * alpha[k][p] * alpha[k][q] * gram_mat
        _third = np.zeros(k, dtype=float)
        for c in range(k):
            _third[c] = np.sum(gram_mat[alpha == c, :][:, alpha == c])
        _third = _third / (num_C ** 2)

        #  k(xj, xj) - 2/|Ck| * sum(alpha_kn * k(xj, xn)) + 1.0 / (k**2) * alpha[k][p] * alpha[k][q] * gram_mat
        new_alpha = np.argmin(_first + _second + _third, axis=1)

        alpha_list.append(new_alpha.reshape(100, 100))

        print('iteration {}, error : {}'.format(iteration, np.sum(np.abs(new_alpha - alpha))))

        if(np.array_equal(alpha, new_alpha)):
            break

        alpha = new_alpha
    
    dir_name = '{}_kernelKmeans_{}_{}'.format(args.img, init, k)
    RGB = plot(alpha_list, k, dir_name)
    

def plot(result, k, dir_name):
    color_mapping = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                              [1, 1, 0], [0, 1, 1], [1, 0, 1],
                              [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5]])
    RGB_result = np.zeros((len(result), 100, 100, 3))
    imgs = []
    for i in range(len(result)):
        RGB_result[i] = color_mapping[result[i]]
        if(not os.path.isdir(dir_name)):
            os.mkdir(dir_name)
        plt.imsave('{}/{}.png'.format(dir_name, i), RGB_result[i])
        imgs.append(Image.fromarray(np.uint8(RGB_result[i] * 255)))
    imgs[0].save('{}/video.gif'.format(dir_name), format='GIF', append_images=imgs[1:], loop=0,
                save_all=True, duration=300)


def KMeans(data, k, cut_type, init='random_point'):
    alpha_list = [] # k-means
    # data.shape (10000, 3)

    # initialization
    if(init == 'random_point'):
        # initialize cluster
        mean_idx = np.random.choice(data.shape[0], size=k)
        means = data[mean_idx]
        print(init)
    elif(init == 'kmeans++'):
        mean_idx = np.random.choice(data.shape[0], size=k)
        means = data[mean_idx]
        for i in range(1, k):
            dis = cdist(data, means, 'euclidean')
            min_dis = np.min(dis, axis=1)
            # compute the probability that each point occurs
            next_mean_idx = np.random.choice(data.shape[0], size=1, p=min_dis/np.sum(min_dis))
            means[i] = data[next_mean_idx]

    # cluster
    dis = cdist(data, means, 'sqeuclidean')
    alpha = np.argmin(dis, axis=1)  # k classes : class 0 ~ k-1     # (10000, 2)

    alpha_list.append(alpha.reshape(100, 100))

    for iteration in range(1, 51):
        # update means
        num_C = np.ones(k, dtype=float)
        for i in range(k):
            tmp = np.count_nonzero(alpha == i)
            if(tmp != 0):
                num_C[i] = tmp
        means = np.zeros((k, k))
        for i in range(k):
            means[i] = np.sum(data[alpha == i, :], axis=0)
            means[i] /= num_C[i]

        # cluster
        dis = cdist(data, means, 'sqeuclidean')
        new_alpha = np.argmin(dis, axis=1)  # k classes : class 0 ~ k-1     # (10000, 2)

        alpha_list.append(new_alpha.reshape(100, 100))
       
        if(np.array_equal(alpha, new_alpha)):
            break

        alpha = new_alpha
    
    dir_name = '{}_spectral_{}_{}_{}'.format(args.img, cut_type, init, k)
    RGB = plot(alpha_list, k, dir_name)

    if(k == 2):
        plotEigenSpace(data, alpha, cut_type)


def normalized(U, n):
    T = U.copy()
    for i in range(n):
        T[i] = U[i] / np.sqrt(np.sum(U[i] ** 2))
    return T

def spectralCluster(data, k=2, cut_type='normalized', init='random_point'):
    
    n = data.shape[0] * data.shape[1]   # 10000
    flat_data = data.reshape(-1, 3) # (10000, 3)
    flat_data = flat_data / 255.0

    # spatial coordinates   # (10000, 2)
    indices = np.array([[i, j] for j in range(data.shape[1]) for i in range(data.shape[0])])
    indices = indices / 99.0

    # initialize W
    W = kernel(indices, flat_data)   # (10000, 10000)
    d = np.sum(W, axis=1)

    if(cut_type == 'normalized'):
        D = np.diag(d)
        D_tmp = inv(np.sqrt(D))
        L = np.matmul( D_tmp, (D-W), D_tmp)  # handle complex numbers
    elif(cut_type == 'ratio'):
        D = np.diag(d)
        L = D - W

    # eigenDecomposition
    eig_val_file = 'eig_val_{}_{}.npy'.format(cut_type, args.img)
    eig_vec_file = 'eig_vec_{}_{}.npy'.format(cut_type, args.img)
    if(os.path.isfile(eig_val_file) and os.path.isfile(eig_vec_file)):
        eig_value = np.load(eig_val_file)
        eig_vector = np.load(eig_vec_file)
    else:
        eig_value, eig_vector = eig(L)
        np.save(eig_vec_file, eig_vector)
        np.save(eig_val_file, eig_value)

    eig_order = np.argsort(eig_value)
    eig_vector = eig_vector[:, eig_order]
    U = eig_vector[:, 1:k+1]   # second to k^th
    if(cut_type == 'normalized'):
        U = normalized(U, n)

    KMeans( U, k, cut_type, init)
    

def plotEigenSpace(U, alpha, cut_type):
    plt.scatter(U[alpha == 0, 0:1], U[alpha == 0, 1:2], c='green')
    plt.scatter(U[alpha == 1, 0:1], U[alpha == 1, 1:2], c='red')
    plt.title('Eigen Space ({})'.format(cut_type))
    plt.savefig("eigen_{}.png".format(cut_type))
    

#################################### main #####################################
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--img", default='image2')
args = parser.parse_args()
img1 = imread(args.img + '.png')

for k in range(2, 5):
    kernelKMeans(img1, k, init='random')
    kernelKMeans(img1, k, init='spatial_nearest')
    kernelKMeans(img1, k, init='kmeans++')
    spectralCluster(img1, k, cut_type='normalized', init='random_point')
    spectralCluster(img1, k, cut_type='normalized', init='kmeans++')
    spectralCluster(img1, k, cut_type='ratio', init='random_point')
    spectralCluster(img1, k, cut_type='ratio', init='kmeans++')

# spectralCluster(img1, 2, cut_type='normalized', init='kmeans++')
# spectralCluster(img1, 2, cut_type='ratio', init='kmeans++')
