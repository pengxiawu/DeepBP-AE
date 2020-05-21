from __future__ import division
from models.model_BPAE import BPAE
from utils.access_deepMIMO_data import datasplit
from utils.utils import LP_BP_avg_err
from scipy import sparse
import os
import numpy as np
import tensorflow as tf

flags = tf.app.flags

flags.DEFINE_string("decoder_type", "GAEC", "choose one from [GAE, GAEC, SAE, SAEC]")
flags.DEFINE_integer('input_dim', 512, "Input dimension [512]")
flags.DEFINE_integer("emb_dim", 9, "Number of measurements [9]")
flags.DEFINE_integer("num_samples", 50000, "Number of total samples [50000]")
flags.DEFINE_integer("decoder_num_steps", 15,
                     "Depth of the decoder network [15]")
flags.DEFINE_integer("batch_size", 128, "Batch size [128]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate for SGD [0.01]")
flags.DEFINE_integer("max_training_epochs", 1000,
                     "Maximum number of training epochs")
flags.DEFINE_integer("display_interval", 5,
                     "Print the training info every [5] epochs")
flags.DEFINE_integer("validation_interval", 5,
                     "Compute validation loss every [5] epochs")
flags.DEFINE_integer("max_steps_not_improve", 1,
                     "stop training when the validation loss \
                      does not improve for [1] validation_intervals")
flags.DEFINE_string("checkpoint_dir", "./results/20200519_deepMIMOdataset_gaec/",
                    "Directory name to save the checkpoints \
                    [./results/]")
flags.DEFINE_integer("num_random_dataset", 1,
                     "Number of random read_result [1]")
flags.DEFINE_integer("num_experiment", 1,
                     "Number of experiments for each datasets [1]")

FLAGS = flags.FLAGS


# model parameters
decoder_type = FLAGS.decoder_type
input_dim = FLAGS.input_dim
emb_dim = FLAGS.emb_dim
num_samples = FLAGS.num_samples
decoder_num_steps = FLAGS.decoder_num_steps
decoder_type = FLAGS.decoder_type

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

    X_train, X_valid, X_test = datasplit(num_samples=num_samples,
                                         train_ratio=0.96, valid_ratio=0.02)
    x = X_train.todense()
    x = np.concatenate((x.clip(min=0), (-x).clip(min=0)), axis=1)
    X_train = sparse.csr_matrix(x)

    x = X_valid.todense()
    x = np.concatenate((x.clip(min=0), (-x).clip(min=0)), axis=1)
    X_valid = sparse.csr_matrix(x)

    x = X_test.todense()
    x = np.concatenate((x.clip(min=0), (-x).clip(min=0)), axis=1)
    X_test = sparse.csr_matrix(x)

    print(np.shape(X_train))
    print(np.shape(X_valid))
    print(np.shape(X_test))

    for experiment_i in range(num_experiment):
        config =  tf.compat.v1.ConfigProto()
        #config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        print("---Dataset: %d, Experiment: %d---" % (dataset_i, experiment_i))
        BP_AE = BPAE(sess, input_dim, emb_dim, decoder_num_steps, decoder_type)

        print("Start training......emb_dim{:02d}".format(emb_dim))
        BP_AE.train(X_train, X_valid, batch_size, learning_rate,
                    max_training_epochs, display_interval,
                    validation_interval, max_steps_not_improve)
        # evaluate the autoencoder
        test_sq_loss = BP_AE.inference(X_test, batch_size)
        print("test_error is: ", test_sq_loss)

        learned_matrix = BP_AE.sess.run(BP_AE.encoder_weight)

        file_name = ('matrix' + 'input_%d_' + 'depth_%d_' + 'emb_%02d.npy') \
                    % (input_dim, decoder_num_steps, emb_dim)
        file_path = checkpoint_dir + file_name
        np.save(file_path, learned_matrix)
        Y = X_test.dot(learned_matrix)
        gaec_lp_err, gaec_lp_exact, gaec_lp_solve = \
            LP_BP_avg_err(np.transpose(learned_matrix), Y, X_test, use_pos=False)
        gaec_lp_err_pos, gaec_lp_exact_pos, gaec_lp_pos_solve = \
            LP_BP_avg_err(np.transpose(learned_matrix), Y, X_test, use_pos=True)


        res = {}
        res['gaec_lp_err'] = gaec_lp_err
        res['gaec_lp_exact'] = gaec_lp_exact
        res['gaec_lp_solve'] = gaec_lp_solve
        res['gaec_lp_err_pos'] = gaec_lp_err_pos
        res['gaec_lp_exact_pos'] = gaec_lp_exact_pos
        res['gaec_lp_pos_solve'] = gaec_lp_pos_solve
        merge_dict(results_dict, res)
        print(res)

# save results_dict
file_name = ('res_'+'input_%d_'+'depth_%d_'+'emb_%02d.npy') \
            % (input_dim, decoder_num_steps, emb_dim)
file_path = checkpoint_dir + file_name
np.save(file_path, results_dict)
