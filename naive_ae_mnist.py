from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

input_img = Input(shape=(784,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(784, activation='sigmoid')(decoded)

autoencoder = Model(input_img, decoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

from keras.datasets import mnist
import numpy as np
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

autoencoder.fit(x_train, x_train,
                epochs=100,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))


# # this is our input placeholder
# input_img = Input(shape=(784,))
# # "encoded" is the encoded representation of the input
# encoded = Dense(encoding_dim, activation='relu',
#                 activity_regularizer=regularizers.l1(10e-5))(input_img)
# # "decoded" is the lossy reconstruction of the input
# decoded = Dense(784, activation='sigmoid')(encoded)
#
# # this models maps an input to its reconstruction
# autoencoder = Model(input_img, decoded)
#
# # this models maps an input to its encoded representation
encoder = Model(input_img, encoded)
#
# # create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(128,))
# # retrieve the last layer of the autoencoder models
# decoder_layer = autoencoder.layers[-1]
decoder_layer = autoencoder.layers[-1]
# # create the decoder models
decoder = Model(encoded_input, decoder_layer(encoded_input))
#
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
#
#
#
#
#
#
#
# autoencoder.fit(x_train, x_train,
#                 epochs=50,
#                 batch_size=256,
#                 shuffle=True,
#                 validation_data=(x_test, x_test))

# encode and decode some digits
# note that we take them from the *test* set
# encoded_imgs = encoder.predict(x_test)
decoded_imgs = autoencoder.predict(x_test)
nmse =  np.linalg.norm(x_test-decoded_imgs)**2/np.linalg.norm(x_test)**2
print('nmse is %f' % nmse)
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()




