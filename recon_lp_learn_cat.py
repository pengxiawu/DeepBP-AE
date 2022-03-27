from __future__ import division
from data.load_data_myGenert import datasplit
from utils import LP_BP_avg_err
from scipy import sparse
import os
import numpy as np
import tensorflow as tf


# model parameters
input_dim = 512 # Input dimension [256]
emb_dim = 32 # Number of measurements [32]
num_samples = 100000 # Number of total samples [100000]

# checkpoint directory
checkpoint_dir = "./res/2021Nov_DeepMIMO_SAEC/" # give the path of the learned matrix

results_dict = {}

def merge_dict(a, b):
    """Merge two dictionaries"""
    for k in b.keys():
        if k in a:
            a[k].append(b[k])
        else:
            a[k] = [b[k]]


_, _, X_test = datasplit(num_samples=num_samples,
                         train_ratio=0.98,
                         valid_ratio=0.01)
x = X_test.todense()
x = np.concatenate((x.clip(min=0), (-x).clip(min=0)), axis=1)
# x = np.concatenate((x, np.zeros_like(x)), axis=1)
X_test = sparse.csr_matrix(x)
print(np.shape(X_test))

learned_matrix = np.load(checkpoint_dir+'matrixinput_512_depth_15_emb_{}.npy'.format(emb_dim))

Y = X_test.dot(learned_matrix)
# noise = 0.005 * np.random.normal(size=np.shape(Y))   # without noise
# Y += noise
# print('SNR is {}'.format(10 * np.log10((np.linalg.norm(X_test.toarray())**2) / ((np.linalg.norm(noise)**2)))))

ae_lp_err, ae_lp_exact, ae_lp_solve = \
            LP_BP_avg_err(np.transpose(learned_matrix), Y, X_test, use_pos=False)
ae_lp_err_pos, ae_lp_exact_pos, ae_lp_solve_pos = \
            LP_BP_avg_err(np.transpose(learned_matrix), Y, X_test, use_pos=True)

res = {}

# res['saec_lp_err'] = ae_lp_err
# res['saec_lp_exact'] = ae_lp_exact
# res['saec_lp_solve'] = ae_lp_solve
# res['saec_lp_err_pos'] = ae_lp_err_pos
# res['saec_lp_exact_pos'] = ae_lp_exact_pos
# res['saec_lp_solve_pos'] = ae_lp_solve_pos

res['gaec_lp_err'] = ae_lp_err
res['gaec_lp_exact'] = ae_lp_exact
res['gaec_lp_solve'] = ae_lp_solve
res['gaec_lp_err_pos'] = ae_lp_err_pos
res['gaec_lp_exact_pos'] = ae_lp_exact_pos
res['gaec_lp_solve_pos'] = ae_lp_solve_pos

print(res)
merge_dict(results_dict, res)


# save results_dict
file_name = ('res_nmse'+'input_%d_'+'emb_%02d.npy') %(input_dim, emb_dim)
file_path = checkpoint_dir + file_name
np.save(file_path, results_dict)
