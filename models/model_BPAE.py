
"""Define the autoencoder"""
from __future__ import division
from time import time
from utils.utils import prepareSparseTensor

import numpy as np
import tensorflow as tf

class BPAE(object):
    def __init__(self, sess, input_dim, emb_dim, decoder_num_steps, decoder_type):
        self.sess = sess
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.decoder_num_steps = decoder_num_steps
        # define the input as a SparseTensor
        self.indices_x = tf.placeholder("int64", [None, 2])
        self.values_x = tf.placeholder("float", [None])
        self.dense_shape_x = tf.placeholder("int64", [2])
        self.input_x = tf.SparseTensor(indices=self.indices_x,
                                       values=self.values_x,
                                       dense_shape=self.dense_shape_x)
        self.encoder_weight = tf.Variable(tf.truncated_normal(
                                    [self.input_dim, self.emb_dim],
                                    stddev=1.0/np.sqrt(self.input_dim)))
        ## encode the input
        self.encode_shape_placeholder = tf.placeholder("int64", [2], name="encode_shape")
        self.encode = tf.sparse_tensor_dense_matmul(self.input_x,
                                                    self.encoder_weight)
        # decode by simulating decoder_num_steps projected subgradient updates
        self.step_size = tf.Variable(1.0)
        self.noise = tf.Variable(0.1)

        def decode_subgrad_bpsaec(x, W, num_steps, step_size):
            x = tf.matmul(x, W, transpose_b=True)
            for i in range(num_steps):
                x = x + (tf.matmul(tf.matmul(tf.sign(x), W), W,
                         transpose_b=True)-tf.sign(x))*(step_size/(i+1))
                x = tf.layers.batch_normalization(x, axis=1)
            x = tf.nn.relu(x)
            return x  # The output layer

        def decode_subgrad_bpsae(x, W, num_steps, step_size):
            x = tf.matmul(x, W, transpose_b=True)
            for i in range(num_steps):
                x = x + (tf.matmul(tf.matmul(tf.sign(x), W), W,
                        transpose_b=True)-tf.sign(x))*(step_size/(i+1))
                x = tf.layers.batch_normalization(x, axis=1)
            x_hat = tf.nn.relu(x) - tf.nn.relu(-x)
            return x_hat

        def decode_subgrad_bpgaec(x, W, num_steps, step_size):
            x = tf.matmul(x, W, transpose_b=True)
            y_t = self.encode
            for i in range(num_steps):
                x = x - tf.matmul(x, (tf.matmul(W, W, transpose_b=True))) \
                    + tf.matmul(y_t, W, transpose_b=True) \
                    + (tf.matmul(tf.matmul(tf.sign(x), W), W,
                                 transpose_b=True) - tf.sign(x)) * (step_size / (i + 1))
                y_t = tf.matmul(x, W)
                x = tf.layers.batch_normalization(x, axis=1)
            x = tf.nn.relu(x)
            return x

        def decode_subgrad_bpgae(x, W, num_steps, step_size):
            x = tf.matmul(x, W, transpose_b=True)
            y_t = self.encode
            for i in range(num_steps):
                x = x - tf.matmul(x, (tf.matmul(W, W, transpose_b=True))) \
                    + tf.matmul(y_t, W, transpose_b=True) \
                    + (tf.matmul(tf.matmul(tf.sign(x), W), W,
                                 transpose_b=True) - tf.sign(x)) * (step_size / (i + 1))
                y_t = tf.matmul(x, W)
                x = tf.layers.batch_normalization(x, axis=1)
            x_hat = tf.nn.relu(x) - tf.nn.relu(-x)
            return x_hat

        if decoder_type== 'GAE':
            self.pred = decode_subgrad_bpgae(self.encode, self.encoder_weight,
                                       self.decoder_num_steps, self.step_size)
        elif decoder_type== 'SAE':
            self.pred = decode_subgrad_bpsae(self.encode, self.encoder_weight,
                                             self.decoder_num_steps, self.step_size)
        elif decoder_type== 'SAEC':
            self.pred = decode_subgrad_bpsaec(self.encode, self.encoder_weight,
                                             self.decoder_num_steps, self.step_size)
        elif decoder_type== 'GAEC':
            self.pred = decode_subgrad_bpgaec(self.encode, self.encoder_weight,
                                             self.decoder_num_steps, self.step_size)

        # define the squared loss
        self.sq_loss = tf.reduce_mean(tf.pow(tf.sparse_add(self.input_x,
                                      -self.pred), 2))*self.input_dim
        self.learning_rate = tf.placeholder("float", [])
        self.sq_optim = tf.train.GradientDescentOptimizer(
                                     self.learning_rate).minimize(self.sq_loss)

    def train(self, X_train, X_valid, batch_size, learning_rate,
              max_training_epochs=2e4, display_interval=1e2,
              validation_interval=10, max_steps_not_improve=5):
        """Perform training on the model
        Args:
            max_training_epochs [1000]: stop training after 1000 epochs.
            display_interval [5]: print the training info every 5 epochs.
            validation_interval [5]: compute validation loss every 5 epochs.
            max_steps_not_improve [5]: stop training when the validation loss
                                does not improve for 5 validation_intervals.
        """
        # initialize the variables
        init = tf.global_variables_initializer()
        self.sess.run(init)
        # early-stopping parameters
        best_valid_loss = self.inference(X_valid, batch_size)
        num_steps_not_improve = 0
        # start training
        t0 = time()
        batch_size = np.amin([batch_size, X_train.shape[0]])
        total_batch = int(X_train.shape[0]/batch_size)
        # training cycle
        current_epoch = 0
        while current_epoch < max_training_epochs:
            train_loss = 0
            # random shuffle
            idx = np.random.permutation(X_train.shape[0])
            # Loop over all batches
            for batch_i in range(total_batch):
                idx_batch_i = idx[batch_i*batch_size:(batch_i+1)*batch_size]
                train = X_train[idx_batch_i, :]
                indices, values, shape = prepareSparseTensor(train)
                # optimize the sq_loss
                _, c = self.sess.run([self.sq_optim, self.sq_loss],
                                     feed_dict={
                                      self.indices_x: indices,
                                      self.values_x: values,
                                      self.dense_shape_x: shape,
                                      self.learning_rate: learning_rate,
                                      self.encode_shape_placeholder: [shape[0], self.emb_dim]})
                train_loss += c
            if current_epoch % validation_interval == 0:
                current_valid_loss = self.inference(X_valid, batch_size)
                if current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    num_steps_not_improve = 0
                else:
                    num_steps_not_improve += 1
            if current_epoch % display_interval == 0:
                # print avg_err,
                print("Epoch: %05d" % (current_epoch),
                      "TrainSqErr: %f" % (train_loss/total_batch),
                      "ValidSqErr: %f" % (current_valid_loss),
                      "StepSize: %f" % (self.sess.run(self.step_size)))
            current_epoch += 1
            # stop training when the validation loss
            # does not improve for certain number of steps
            if num_steps_not_improve > max_steps_not_improve:
                break
        print("Optimization Finished!")
        t1 = time()
        print("Training takes %d epochs in %f secs" % (current_epoch, t1-t0))
        print("Training loss: %f" % (train_loss/total_batch))
        print("Validation loss: %f" % (current_valid_loss))

    def inference(self, X, batch_size):
        """Perform inference on the model"""
        batch_size = np.amin([batch_size, X.shape[0]])
        total_batch = int(X.shape[0]/batch_size)
        total_loss = 0
        # loop over all batches
        for batch_i in range(total_batch):
            inputs = X[batch_i*batch_size:(batch_i+1)*batch_size, :]
            indices, values, shape = prepareSparseTensor(inputs)
            # get the loss value
            c = self.sess.run(self.sq_loss, feed_dict={
                                            self.indices_x: indices,
                                            self.values_x: values,
                                            self.dense_shape_x: shape,
                                            self.encode_shape_placeholder: [shape[0], self.emb_dim]})
            total_loss += c
        return total_loss/total_batch
