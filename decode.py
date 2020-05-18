# test the LP decoder using Gaussian measurement matrix

from __future__ import division
from time import time
from my_datasets import datasplit
from baselines import l1_min

import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_integer('input_dim', 512, "Input dimension [1000]")
flags.DEFINE_integer("emb_dim", 70, "Number of measurements [10]")
flags.DEFINE_integer("num_samples", 10000, "Number of total samples [10000]")
flags.DEFINE_integer("decoder_num_steps", 10,
                     "Depth of the decoder network [10]")

FLAGS = flags.FLAGS

input_dim = FLAGS.input_dim
emb_dim = FLAGS.emb_dim
num_samples = FLAGS.num_samples
decoder_num_steps = FLAGS.decoder_num_steps

_, _, X_test = datasplit(num_samples=10000, train_ratio=0.6, valid_ratio=0.2)

# LP optimization decoder
print("Start l1-min......")
t0 = time()
res = l1_min(X_test, input_dim, emb_dim)
t1 = time()
print("L1-min takes %f secs") % (t1 - t0)
print(res)