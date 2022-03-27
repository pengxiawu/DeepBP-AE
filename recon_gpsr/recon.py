from __future__ import division
import scipy.io as sio
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import time
import matplotlib
from functions_def import grad_proj_descent_BBmono, DC_GPSR_2loopBBmono,\
     lasso_gradient, proj_nonnegative


"""
    PLOT
"""
kwargs = {'linewidth': 3.0, 'linestyle': '-'}
font = {'weight': 'normal', 'size': 12}
matplotlib.rc('font', **font)


def error_plot(ys, label, yscale='log'):
    plt.yscale(yscale)
    plt.plot(range(len(ys)), ys, label=label, **kwargs)
    plt.xlim(left=0)


kwargs_scatter = {'s': 80, 'c': 'r', 'marker': 'o'}


def error_scatter(xidx, ys, label, yscale='log'):
    plt.yscale(yscale)
    plt.scatter(xidx, ys, label=label, **kwargs_scatter)
    plt.xlim(left=0)


np.random.seed(1337)

# matdata = '../data/H_beam_sparsity16(deepMIMO).mat'
# dataname = 'H_beam_sparsity_syn'

matdata = '../data/2020otc_H_beam_sparsity_syn16.mat'
dataname = 'H_beam_sparsity_syn'


matdata = sio.loadmat(matdata)
K = matdata[dataname].T
X = np.array(K)
# X = normalize(X, norm='l2', axis=1, copy=False, return_norm=False)

sam_ind = 0
# sam_ind = int(len(X)/2-2)
s = np.concatenate((X[2*sam_ind, :], X[2*sam_ind+1, :])).ravel()
print(np.linalg.norm(s))
print(s.shape)
print(np.linalg.norm(s))
print(np.count_nonzero(s[np.absolute(s)>0]))  # 32

f = np.concatenate((s.clip(min=0), (-s).clip(min=0)), axis=0)


n = len(s)

emb_dim = 32
Phi_g = np.random.randn(emb_dim, n//2)
Phi_b = np.random.binomial(1, 0.5, size=(emb_dim, n//2))
Phi_learned = np.load(
    '../res/2020otc_MyGenerateDataset_gae(1e5samples_noisy)/'
    'matrixinput_256_depth_15_emb_{}.npy'.format(emb_dim)).transpose()

Phi = normalize(Phi_learned, axis=0) # choose leanred matrix or a random matrix
A1 = np.hstack((Phi, np.zeros(shape=np.shape(Phi))))
A2 = np.hstack((np.zeros(shape=np.shape(Phi)), Phi))
A = np.vstack((A1, A2))
R = np.concatenate((A, -A), axis=1)
b = R.dot(f)
# noise = 0.02* np.random.normal(size=np.size(b))   # without noise
# b += noise
# print('SNR is {}'.format(10*np.log10(1.0/(np.linalg.norm(noise)**2))))


x0 = np.zeros(s.shape)
f0 = np.zeros(f.shape)


n_spikes = 3  # the value of K in K-norm for DC-GPSR algorithm
k = 1 # compressed dimension, but only used as a constant when computing objective
penalty_dc = 0.001
step_size = 1.0

penalty_lasso = 0.001


"""
    double-loop DC-GPSR-BB for sparse reconstruction
"""
ilop = 1000

# for double axeses setting
def overall2out(x):
    return x / ilop

def out2overall(x):
    return x * ilop

start_time = time.time()

fs_dc2_proj_descent, \
outer_loop_fs, \
outer_idx = DC_GPSR_2loopBBmono(init=f0,
                                inner_steps_list=[step_size] * ilop,
                                grad=lambda x: lasso_gradient(R, b, k, x, penalty_dc),
                                proj=proj_nonnegative,
                                outer_loop_num = 20,
                                sparsity=n_spikes,
                                penalty=penalty_dc)

end_time = time.time()
print("Double-loop DC-GPSR running time --- %s seconds ---" % (end_time - start_time))


"""
    GPSR-BB
"""

start_time = time.time()
fs_proj_grad = grad_proj_descent_BBmono(f0,
                                        [step_size]*20000,
                                        lambda x: lasso_gradient(R, b, k, x, penalty_lasso),
                                        proj=proj_nonnegative)[0]
end_time = time.time()
print("l1-regularized GPSR running time --- %s seconds ---" % (end_time - start_time))


"""
    Reconstruct and accuracy performances
"""
xs_dc2_proj_descent = fs_dc2_proj_descent[-1][:n] - fs_dc2_proj_descent[-1][n::]
nmse_dc2_proj_descent = np.linalg.norm(s - xs_dc2_proj_descent)**2 / (np.linalg.norm(s)**2)
# mse_dc1_proj_descent = np.linalg.norm(s - xs_dc1_proj_descent)**2
print("The double-loop DC-GPSR NMSE is : {}".format(nmse_dc2_proj_descent))
# print("The double-loop DC-GPSR MSE is : {}".format(mse_dc1_proj_descent))

xs_proj_grad = fs_proj_grad[-1][:n] - fs_proj_grad[-1][n::]
nmse_proj_grad = np.linalg.norm(s - xs_proj_grad)**2 / (np.linalg.norm(s)**2)
# mse_proj_grad = np.linalg.norm(s - xs_proj_grad)**2
print("The l1-regularized GPSR NMSE is : {}".format(nmse_proj_grad))
# print("The Lasso-GPSR MSE is : {}".format(mse_proj_grad))

"""complex channels"""
s_comp = s[:int(n/2)] + 1j * s[int(n/2)::]
s_rec_dc2_comp = xs_dc2_proj_descent[:int(n/2)] + 1j*xs_dc2_proj_descent[int(n/2)::]
s_rec_gpsr_comp = xs_proj_grad[:int(n/2)] + 1j*xs_proj_grad[int(n/2)::]


"""
    Fig.1
    Illustration of the reconstruction signals and the optimal signals
    # plt.title('Comparison of reconstructed signal and optimal signal')
    # plt.title('Comparison of initial, optimal, and computed point')
"""
font = {'weight': 'normal', 'size': 25}
matplotlib.rc('font', **font)
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
plt.figure(figsize=(20, 5))
idxs = range(len(s_comp))
markerline, stemlines, baseline = plt.stem(idxs, np.abs(s_comp[idxs]),
                                           label='Optimal')
plt.setp(markerline, 'markersize', 5)
plt.scatter(idxs, np.abs(s_rec_dc2_comp[idxs]),
            label='DC-GPSR reconstruct with learned matrix $\mathbf{\Phi}_{gae}$',
            # label='DC-GPSR reconstruct with Gaussian matrix $\mathbf{G}$',
            s=80, color='m', marker='D')
plt.xlabel('Antenna Index')
plt.ylabel('Channel Magnitude')
plt.xlim(0, len(s_comp))
plt.legend(loc='upper left', prop={'size': 25})
plt.text(5.0, 0.25,
         'Normalized squared $\ell_2$-error: %.2e'
         %(nmse_dc2_proj_descent),
         fontsize=25)
plt.tight_layout()
plt.savefig(__file__.rstrip('.py').split('/')[-1] + "_fig-1.eps", format='eps')
plt.show()

"""
    Fig.2
    Illustration of the reconstruction signals and the optimal signals
    # plt.title('Comparison of reconstructed signal and optimal signal')
    # plt.title('Comparison of initial, optimal, and computed point')
"""
font = {'weight': 'normal', 'size': 25}
matplotlib.rc('font', **font)
plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = True
plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = False
plt.figure(figsize=(20, 5))
idxs = range(len(s_comp))
markerline, stemlines, baseline = plt.stem(idxs, np.abs(s_comp[idxs]),
                                           label='Optimal')
plt.setp(markerline, 'markersize', 5)
plt.scatter(idxs, np.abs(s_rec_gpsr_comp[idxs]),
            label='$\ell_1$-GPSR Reconstruct with learned matrix $\mathbf{\Phi}_{gae}$',
            # label='$\ell_1$-GPSR reconstruct with Gaussian matrix $\mathbf{G}$',
            s=80, color='m', marker='D')
plt.xlabel('Antenna Index')
plt.ylabel('Channel Magnitude')
plt.xlim(0, len(s_comp))
plt.legend(loc='upper left', prop={'size': 25})
plt.text(5.0, 0.25,
         'Normalized squared $\ell_2$-error: %.2e'
         % (nmse_proj_grad,),
         fontsize=25)
plt.tight_layout()
plt.savefig(__file__.rstrip('.py').split('/')[-1] + "_fig-2.eps", format='eps')
plt.show()
