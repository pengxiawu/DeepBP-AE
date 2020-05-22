from __future__ import division
import scipy.sparse as sp
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import normalize


"""load datasets"""
matdata1='../datasets/my_generate_K13p2dB/H_beam_sparsity_syn3.mat'
dataname1='H_beam_sparsity_syn'
matdata1 = sio.loadmat(matdata1)



K = matdata1[dataname1].T
X = np.array(K)
np.random.seed(43)
shuffle = np.random.permutation(X.shape[0])
X = X[shuffle, :]
X = sp.csc_matrix(X)
X = normalize(X, norm='l2', axis=1, copy=False, return_norm=False)


def datasplit(num_samples, train_ratio, valid_ratio):
    train_size = int(train_ratio*num_samples)
    valid_size = int(valid_ratio*num_samples)
    X_train = X[:train_size, :]
    X_valid = X[train_size:(train_size + valid_size), :]
    X_test = X[(train_size + valid_size):num_samples, :]
    return X_train, X_valid, X_test

