from __future__ import division
from data.load_data_myGenert import datasplit
from utils import LP_BP_avg_err
import numpy as np


# model parameters
input_dim = 256 # Input dimension [256]
emb_dim = 32 # Number of measurements [32]
num_samples = 100000 # Number of total samples [100000]

# checkpoint directory
checkpoint_dir = "./res/2021Nov_DeepMIMO_SAE/" # give the path of the learned matrix

results_dict = {}

def merge_dict(a, b):
    """Merge two dictionaries"""
    for k in b.keys():
        if k in a:
            a[k].append(b[k])
        else:
            a[k] = [b[k]]

_, _, X_test = datasplit(num_samples=num_samples,
                        train_ratio=0.98, valid_ratio=0.01)

learned_matrix = np.load(checkpoint_dir+'matrixinput_256_depth_15_emb_{}.npy'.format(emb_dim))

Y = X_test.dot(learned_matrix)
# noise = 0.005 * np.random.normal(size=np.shape(Y))   # without noise
# Y += noise
# print('SNR is {}'.format(10 * np.log10((np.linalg.norm(X_test.toarray())**2) / ((np.linalg.norm(noise)**2)))))

ae_lp_err, ae_lp_exact, ae_lp_solve = \
            LP_BP_avg_err(np.transpose(learned_matrix), Y, X_test, use_pos=False)

res = {}
res['sae_lp_err'] = ae_lp_err
res['sae_lp_exact'] = ae_lp_exact
res['sae_lp_solve'] = ae_lp_solve
# res['gae_lp_err'] = ae_lp_err
# res['gae_lp_exact'] = ae_lp_exact
# res['gae_lp_solve'] = ae_lp_solve
print(res)
merge_dict(results_dict, res)


# save results_dict
file_name = ('res_nmse_'+'input_%d_'+'emb_%02d.npy') %(input_dim, emb_dim)
file_path = checkpoint_dir + file_name
np.save(file_path, results_dict)
