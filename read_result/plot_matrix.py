import numpy as np
from matplotlib import pyplot as plt
import os
from sklearn.preprocessing import normalize

# """draw data-driven matrices"""
# matrix_path = "../results/20200519_deepMIMOdataset_sae_cat0"
# matrix_name = "matrixinput_512_depth_10_emb_15"
# matrix = np.load(os.path.join(matrix_path, matrix_name)+".npy")

"""draw random Gaussian matrix"""
matrix_path = "../results/20200519_deepMIMOdataset_lp_random_cat0"
matrix_name = "Gaussian_15x512"
matrix = np.load(os.path.join(matrix_path, matrix_name)+".npy")


print(matrix.shape, type(matrix))
plt.plot(np.arange(matrix.shape[1]), matrix[250], 'k-o', linestyle = '--', label="Elements in the 250th column")
plt.plot(np.arange(matrix.shape[1]), matrix[260], 'k->', linestyle = '-', label='Elements in the 260th column')
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.savefig(os.path.join(matrix_path, matrix_name)+'_250_260_columns_l1ae')
plt.show()

fig = plt.imshow(np.transpose(matrix), cmap="Greys")
plt.axis('on')
plt.savefig(os.path.join(matrix_path, matrix_name))
plt.show()

fig2 = plt.imshow(np.transpose(matrix[256-10:256+11,:]), cmap="Greys")
plt.xticks(np.arange(0, 21, step=10), ['246', '256', '266'], fontsize=18)
plt.yticks(fontsize=18)
plt.axis('on')
plt.savefig(os.path.join(matrix_path, matrix_name)+'_zoom')
plt.show()


