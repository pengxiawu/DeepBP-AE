from __future__ import division
from time import time
from data.load_data_myGenert import datasplit
from baselines import LP_BP
import os
import numpy as np

# model parameters
input_dim = 256 # Input dimension [256]
emb_dim = 32 # Number of measurements [32]
num_samples = 100000 # Number of total samples [100000]

# checkpoint directory
checkpoint_dir = "./res/2021Nov_DeepMIMO_lp_random(nmse)/"
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

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

print(np.shape(X_test))

res = {}

# l1 minimization
print("Start LP_BP......")
t0 = time()
res = LP_BP(X_test, input_dim, emb_dim)
t1 = time()
print("LP_BP takes {} sec.".format(t1 - t0))
merge_dict(results_dict, res)
print(res)

# save results_dict
file_name = ('res_'+'input_%d_'+'emb_%02d.npy') %(input_dim, emb_dim)
file_path = checkpoint_dir + file_name
np.save(file_path, results_dict)
