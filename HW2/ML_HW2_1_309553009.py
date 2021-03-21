import gzip
import numpy as np
import math

def loadMNIST(data_file, label_file):

    f_ti = gzip.open(data_file, 'rb')
    f_tl = gzip.open(label_file, 'rb')

    _ = f_ti.read(4)  # magic_number(4 bytes)
    img_num = int.from_bytes(f_ti.read(4), "big")
    rows = int.from_bytes(f_ti.read(4), "big")
    cols = int.from_bytes(f_ti.read(4), "big")

    _ = f_tl.read(8)  # magic_number(4 bytes), item number(4 bytes)

    img_pixels = []
    img_label = []
    for n in range(img_num):
        pixels = f_ti.read(rows * cols)
        label = int.from_bytes(f_tl.read(1), "big")

        img_pixels.append(pixels)
        img_label.append(label)

    f_ti.close()
    f_tl.close()

    return img_pixels, img_label



def getTrainProb(train_x, train_y):
    total_class = 10
    rows = 28
    cols = 28
    total_px = rows * cols
    total_bin = 32

    ## ex: P(y=0|pixels[0]=0) = P(y=0)*P(pixels[0]=0|y=0) / P(pixels[0]=0)

    ## prior --> count each class
    class_num = np.zeros(total_class, dtype=float)
    for lb in train_y:
        class_num[lb] = class_num[lb] + 1
    prior = class_num / float(len(train_y))

    ## likelihood --> count each class on each pixel with 32 possible values 
    class_bin_px_num = np.zeros([total_class, total_px, total_bin], dtype=float)
    for img_idx, lb in enumerate(train_y):
        for px_idx, px in enumerate(train_x[img_idx]):
            class_bin_px_num[int(lb)][px_idx][int(px)] += 1
  
    likelihood = np.zeros([total_class, total_px, total_bin])
    for cls in range(total_class):
        for px_idx in range(total_px):
            for bin in range(total_bin):
                if(class_num[lb] != 0):
                    likelihood[cls][px_idx][bin] = float(class_bin_px_num[cls][px_idx][bin] / class_num[cls])
                else:
                    likelihood[cls][px_idx][bin] = 1e-8

    return prior, likelihood



def NaiveBayesClassifierDiscrete(train_x, train_y, test_x):
    total_class = 10

    ########## train: get prior, likelihood, marginal from train set
    prior, likelihood = getTrainProb(train_x, train_y)

    ########## test: calculate the posterior of the test set
    posterior_list = []
    for n, img_px in enumerate(test_x):

        ## N^th test image, calculate the posterior of class 0 ~ 9
        posterior = np.zeros(total_class, dtype=float)
        for c in range(0, 10):      # class 0 ~ 9            
            log_likelihood = 0.0
            for idx, px in enumerate(img_px):
                log_likelihood = log_likelihood + math.log(max(1e-4, likelihood[c][idx][int(px)]))
            posterior[c] = math.log(max(1e-4, prior[c])) + log_likelihood
        posterior = posterior / np.sum(posterior)

        posterior_list.append(posterior)
    
    return posterior_list, likelihood


def printResult(posterior, test_y, toggle, likelihood_or_mean):

    # print posterior first
    error_rate = 0.0

    for n in range(len(posterior)):
        print('Posterior (in log scale):')
        for c, prob in enumerate(posterior[n]):
            print('{}: {}'.format(c, prob))
        pred = np.argmin(posterior[n])
        print('Prediction: {}, Ans: {}\n'.format(pred, test_y[n]))
        if(pred != test_y[n]):
            error_rate += 1

    # print the imagination of numbers
    printImagination(toggle, likelihood_or_mean)

    # print error rate
    print('Error rate: {}'.format(error_rate / len(test_y)))



def tallyFreq(x):
    new_x = []
    for cur_x in x:
        new_img_px = []
        for i in range(len(cur_x)):
            new_img_px.append(int(cur_x[i]) / 8)  # 256 bins to 32 bins
        new_x.append(new_img_px)
    return new_x


def DiscreteMode(train_x, train_y, test_x, test_y):

    ## tally the frequency
    new_train_x = tallyFreq(train_x)
    new_test_x = tallyFreq(test_x)

    ## Naive Bayes Classifier
    posterior, likelihood = NaiveBayesClassifierDiscrete(new_train_x, train_y, new_test_x)
    
    printResult(posterior, test_y, 0, likelihood)

def printImagination(toggle, likelihood_or_mean):
    total_class = 10
    cols = 28
    rows = 28
    total_px = cols * rows

    if(toggle == 0):    # discrete mode
        likelihood = likelihood_or_mean
        guess = np.zeros([total_class, total_px])
        for c in range(total_class):
            for px in range(total_px):
                guess[c][px] = np.argmax(likelihood[c][px][:]) * 8     # 32 --> 256

    else:               # continuous mode
        guess = likelihood_or_mean  # mean
    
    for c in range(total_class):
        print('{}:'.format(c))
        for px_idx in range(total_px):
            if(px_idx % rows == cols-1):
                print('\n')
            else:
                print('{}'.format( int((guess[c][px_idx]) >= 128) ), end=' ')
                

def NaiveBayesClassifierContinuous(test_x, prior, mean, variance):
    total_class = 10

    posterior_list = []
    for n, px_list in enumerate(test_x):

        posterior = np.zeros(total_class)
        for c in range(total_class):

            posterior[c] += math.log(prior[c])
            for px_idx, px in enumerate(px_list):
                if(variance[c][px_idx] == 0.0): # prevent from divide by 0
                    variance[c][px_idx] = 1e-8
                g = math.log(1.0 / (math.sqrt(2.0 * math.pi * variance[c][px_idx]))) - ((px - mean[c][px_idx])**2 / (2.0 * variance[c][px_idx]))
                posterior[c] += g
    
        posterior = posterior / np.sum(posterior)       # marginalize
        posterior_list.append(posterior)
    
    return posterior_list


def ContinuousMode(train_x, train_y, test_x, test_y):

    ## MLE
    prior, mean, variance = MLE(train_x, train_y)

    ## Naive Bayes Classifier
    posterior = NaiveBayesClassifierContinuous(test_x, prior, mean, variance)

    printResult(posterior, test_y, 1, mean)


def MLE(train_x, train_y):
    total_class = 10
    total_px = 28 * 28

    class_num = np.zeros(total_class, dtype=float)
    for lb in train_y:
        class_num[lb] = class_num[lb] + 1
    prior = class_num / float(len(train_y))

    ## mean & variance
    class_px_sum = np.zeros([total_class, total_px])
    square = np.zeros([total_class, total_px])
    for n, lb in enumerate(train_y):
        for px_idx in range(len(train_x[n])):
            class_px_sum[lb][px_idx] += train_x[n][px_idx]
            square[lb][px_idx] += (train_x[n][px_idx] ** 2)

    variance = np.zeros([total_class, total_px])
    mean = np.zeros([total_class, total_px])
    for c in range(total_class):
        for px_idx in range(total_px):
            mean[c][px_idx] = float(class_px_sum[c][px_idx] / class_num[c])
            # Var(X) = E(X^2) -  E(X)^2 
            # add 1200 to enlarge the variance
            variance[c][px_idx] = float(square[c][px_idx] / class_num[c]) - float(mean[c][px_idx] ** 2) + 1200

    return prior, mean, variance


######################## main #########################

#### read train and test data
train_img_file = 'train-images-idx3-ubyte.gz'
train_label_file = 'train-labels-idx1-ubyte.gz'
test_img_file = 't10k-images-idx3-ubyte.gz'
test_label_file = 't10k-labels-idx1-ubyte.gz'

train_pixels, train_labels = loadMNIST(train_img_file, train_label_file)
test_pixels, test_labels = loadMNIST(test_img_file, test_label_file)

option = int(input('Toggle option: '))

if(option == 0):    # Discrete Mode
    DiscreteMode(train_pixels, train_labels, test_pixels, test_labels)
else:
    ContinuousMode(train_pixels, train_labels, test_pixels, test_labels)
