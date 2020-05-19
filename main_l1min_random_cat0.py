from __future__ import division
from time import time
from utils.my_datasets import datasplit
from scipy import sparse
from utils.baselines import l1_min
import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('input_dim', 512, "Input dimension [512]")
flags.DEFINE_integer("emb_dim", 15, "Number of measurements [10]")
flags.DEFINE_integer("num_samples", 50000, "Number of total samples [10000]")
flags.DEFINE_string("checkpoint_dir", "/home/lab2255/Myresult/csic_res/RES/20200517_deepMIMOdataset_l1min_random_cat0/",
                    "Directory name to save the checkpoints \
                    [RES/cl_res/]")
flags.DEFINE_integer("num_random_dataset", 1,
                     "Number of random read_result [1]")
flags.DEFINE_integer("num_experiment", 1,
                     "Number of experiments for each dataset [1]")

FLAGS = flags.FLAGS


# models parameters
input_dim = FLAGS.input_dim
emb_dim = FLAGS.emb_dim
num_samples = FLAGS.num_samples
decoder_num_steps = FLAGS.decoder_num_steps

# training parameters
batch_size = FLAGS.batch_size
learning_rate = FLAGS.learning_rate
max_training_epochs = FLAGS.max_training_epochs
display_interval = FLAGS.display_interval
validation_interval = FLAGS.validation_interval
max_steps_not_improve = FLAGS.max_steps_not_improve

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
    x = X_test.todense()
    x = np.concatenate((x, np.zeros_like(x)), axis=1)
    X_test = sparse.csr_matrix(x)

    print(np.shape(X_test))

    res = {}

    # l1 minimization
    print("Start l1-min......")
    t0 = time()
    res = l1_min(X_test, input_dim, emb_dim)
    t1 = time()
    print("L1-minimization takes {} sec.".format(t1 - t0))
    merge_dict(results_dict, res)
    print(res)


# save results_dict
file_name = ('resl1min_'+'input_%d_'+'depth_%d_'+'emb_%02d.npy') \
            % (input_dim, decoder_num_steps, emb_dim)
file_path = checkpoint_dir + file_name
np.save(file_path, results_dict)
