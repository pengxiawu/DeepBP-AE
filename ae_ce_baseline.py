from keras.layers import Input, Dense
from keras.models import Model
from my_datasets import datasplit
import numpy as np
from keras import regularizers
import matplotlib.pyplot as plt
import tensorflow as tf


encoding_dim = 15
num_samples = 40000
input_dim = 512

# this is our input placeholder
input_cv = Input(shape=(input_dim,))
# "encoded" is the encoded representation of the input
#encoded = Dense(256, activation='relu')(input_cv)
encoded = Dense(128, activation='relu')(input_cv)
#encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(encoding_dim, activation='relu')(encoded)
# "decoded" is the lossy reconstruction of the input
#decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(encoded)
#decoded = Dense(256, activation='relu')(decoded)
decoded = Dense(input_dim, activation='relu')(decoded)


# this models maps an input to its reconstruction
autoencoder = Model(input_cv, decoded)
# this models maps an input to its encoded representation
encoder = Model(input_cv, encoded)
# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(128,))
# retrieve the last layer of the autoencoder models
decoder_layer = autoencoder.layers[-1]
# create the decoder models
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adam', loss='mse')

X_train, _, X_test = datasplit(num_samples=num_samples,
                                train_ratio=0.8, valid_ratio=0.1)
x = X_train.todense()
x_train = np.concatenate((x.clip(min=0), (-x).clip(min=0)), axis=1)

x = X_test.todense()
x_test = np.concatenate((x.clip(min=0), (-x).clip(min=0)), axis=1)


# print(x_train.shape, x_test.shape) #(32000, 512) (4000, 512)

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


#encoded_cvs = encoder.predict(x_test)
#print(x_test.shape)
#decoded_cvs = decoder.predict(encoded_cvs)
#print(decoded_cvs.shape)
decoded_cvs = autoencoder.predict(x_test)
nmse =  np.linalg.norm(decoded_cvs-x_test)**2 / np.linalg.norm(x_test)**2
print('Average nmse is %f' % nmse)


s_hat = np.reshape(decoded_cvs[10, :], [512])
s = np.reshape(np.array(x_test[10, :]), [512])
plt.figure(1)
plt.plot(s_hat, color='r', marker='*')
plt.figure(2)
plt.show()
plt.plot(s, color='b', marker='o', linestyle='-.')
plt.show()

nmse_s = np.linalg.norm(s_hat-s)**2 / np.linalg.norm(s)**2
print('Sample nmse is %f' % nmse_s)

