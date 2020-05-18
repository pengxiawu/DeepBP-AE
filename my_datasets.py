from __future__ import division
import scipy.sparse as sp
import scipy.io as sio
import numpy as np
import theano
from theano import sparse
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import normalize

# """The dataset used in the csic_full paper"""
# matdata1 = '/home/lab2255/Mydataset/csic_l1ae_dataset/H_beam_sparsity3_0403.mat'
# dataname1='H_beam_sparsity3_syn'
# matdata1 = sio.loadmat(matdata1)


"""alternative dataset"""
# matdata1 = '/home/lab2255/Mydataset/sparsity16_04302020/H_beam_sparsity16_syn.mat'
matdata1 ='/home/lab2255/Mydataset/deepMIMO_dataset_2020May15/H_beam_sparsity_syn3.mat'
# dataname1 = 'H_beam_cut_syn'   %% for /home/lab2255/Mydataset/csic_l1ae_dataset/H_beam_sparsity6_sam20000.mat
# dataname1 = 'H_beam_sparsity16_syn'
# dataname1='H_beam_sparsity3_syn'  ## for /home/lab2255/Mydataset/csic_l1ae_dataset/H_beam_sparsity3_0403.mat
dataname1='H_beam_sparsity_syn'
# dataname1= 'H_beam_sparse_cat'
matdata1 = sio.loadmat(matdata1)



K = matdata1[dataname1].T
X = np.array(K)
np.random.seed(43)
shuffle = np.random.permutation(X.shape[0])
X = X[shuffle, :]
X = sp.csc_matrix(X)

X = normalize(X, norm='l2', axis=1, copy=False, return_norm=False)

# transformer = MaxAbsScaler().fit(X)
# X = transformer.transform(X)
# x = sparse.csc_matrix(name='x', dtype='float64')
# y = sparse.structured_add(x, 1)
# f = theano.function([x], y)
# X = f(X) / 2

def datasplit(num_samples, train_ratio, valid_ratio):
    train_size = int(train_ratio*num_samples)
    valid_size = int(valid_ratio*num_samples)
    X_train = X[:train_size, :]
    X_valid = X[train_size:(train_size + valid_size), :]
    X_test = X[(train_size + valid_size):num_samples, :]
    return X_train, X_valid, X_test

