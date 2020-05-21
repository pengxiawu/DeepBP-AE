from __future__ import division
from time import time
from utils.access_deepMIMO_data import datasplit
from utils.baselines import LP_BP
import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('input_dim', 256, "Input dimension [512]")
flags.DEFINE_integer("emb_dim", 15, "Number of measurements [15]")
flags.DEFINE_integer("num_samples", 50000, "Number of total samples [50000]")
flags.DEFINE_string("checkpoint_dir", "./results/20200519_deepMIMOdataset_lp_random/",
                    "Directory name to save the checkpoints \
                    [./results/]")
flags.DEFINE_integer("num_random_dataset", 1,
                     "Number of random read_result [1]")
flags.DEFINE_integer("num_experiment", 1,
                     "Number of experiments for each datasets [1]")

FLAGS = flags.FLAGS


# models parameters
input_dim = FLAGS.input_dim
emb_dim = FLAGS.emb_dim
num_samples = FLAGS.num_samples

# checkpoint directory
checkpoint_dir = FLAGS.checkpoint_dir
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# number of experiments
num_random_dataset = FLAGS.num_random_dataset
num_experiment = FLAGS.num_experiment

results_dict = {}


def merge_dict(a, b):
    """Merge two dictionaries"""
    for k in b.keys():
        if k in a:
            a[k].append(b[k])
        else:
            a[k] = [b[k]]


for dataset_i in range(num_random_dataset):

    _, _, X_test = datasplit(num_samples=num_samples,
                             train_ratio=0.96,
                             valid_ratio=0.02)

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
file_name = ('res_'+'input_%d_'+'emb_%02d.npy') \
            % (input_dim, emb_dim)
file_path = checkpoint_dir + file_name
np.save(file_path, results_dict)
