from __future__ import division
from utils.utils import LP_BP_avg_err
import numpy as np



def LP_BP(X, input_dim, emb_dim):
    """
    Args:
        X: csr_matrix, shape=(num_sample, input_dim)
    """
    # random Gaussian matrix
    G = np.random.randn(input_dim, emb_dim) / np.sqrt(input_dim)
    Y = X.dot(G) # sparse.csr_matrix.dot
    g_err, g_exact, _ = LP_BP_avg_err(np.transpose(G), Y, X, use_pos=False)
    g_err_pos, g_exact_pos, _ = LP_BP_avg_err(np.transpose(G), Y, X, use_pos=True)


    # random Selection (0/1) matrix
    S = np.random.binomial(1, 0.5, (input_dim, emb_dim))/np.sqrt(emb_dim)
    Y = X.dot(S)  # sparse.csr_matrix.dot
    s_err, s_exact, _ = LP_BP_avg_err(np.transpose(S), Y, X, use_pos=False)
    s_err_pos, s_exact_pos, _ = LP_BP_avg_err(np.transpose(S), Y, X, use_pos=True)

    # random Bernoulli(-1/1) matrix
    B = (np.random.binomial(1, 0.5, (input_dim, emb_dim))*2-1) / np.sqrt(emb_dim)
    Y = X.dot(B)  # sparse.csr_matrix.dot
    b_err, b_exact, _ = LP_BP_avg_err(np.transpose(B), Y, X, use_pos=False)
    b_err_pos, b_exact_pos, _ = LP_BP_avg_err(np.transpose(B), Y, X, use_pos=True)

    # random discrete Fourier transform matrix
    F = np.zeros((input_dim, emb_dim))
    # select emb_dim/2 rows since the DFT matrix is complex
    for col in range(int(emb_dim/2)):
        k = np.random.choice(input_dim, 1)
        for row in range(input_dim):
            F[row, 2*col] = np.cos(-(row*k*2*np.pi)/input_dim)
            F[row, 2*col+1] = np.sin(-(row*k*2*np.pi)/input_dim)
    F = F/np.sqrt(emb_dim/2)
    Y = X.dot(F)  # sparse.csr_matrix.dot
    f_err, f_exact, _ = LP_BP_avg_err(np.transpose(F), Y, X, use_pos=False)
    f_err_pos, f_exact_pos, _ = LP_BP_avg_err(np.transpose(F), Y, X, use_pos=True)

    # random phase shifter transform matrix
    P = np.zeros((input_dim, emb_dim))
    # define the quantized angles
    theta = np.zeros((1, input_dim))
    for i in range(int(input_dim)):
        theta[0, i] = (i*2*np.pi)/input_dim
    for col in range(int(emb_dim/2)):
        for row in range(int(input_dim)):
            u = np.random.choice(input_dim, 1)
            P[row, 2*col] = np.cos(-theta[0, u])
            P[row, 2*col] = np.sin(-theta[0, u])
    P = P/np.sqrt(emb_dim/2)
    Y = X.dot(P)
    p_err, p_exact, _ = LP_BP_avg_err(np.transpose(P), Y, X, use_pos=False)
    p_err_pos, p_exact_pos, _ = LP_BP_avg_err(np.transpose(P), Y, X, use_pos=True)

    res = {}
    res['l1_g_err'] = g_err
    res['l1_g_exact'] = g_exact
    res['l1_g_err_pos'] = g_err_pos
    res['l1_g_exact_pos'] = g_exact_pos
    res['l1_s_err'] = s_err
    res['l1_s_exact'] = s_exact
    res['l1_s_err_pos'] = s_err_pos
    res['l1_s_exact_pos'] = s_exact_pos
    res['l1_b_err'] = b_err
    res['l1_b_exact'] = b_exact
    res['l1_b_err_pos'] = b_err_pos
    res['l1_b_exact_pos'] = b_exact_pos
    res['l1_f_err'] = f_err
    res['l1_f_exact'] = f_exact
    res['l1_f_err_pos'] = f_err_pos
    res['l1_f_exact_pos'] = f_exact_pos
    res['l1_p_err'] = p_err
    res['l1_p_exact'] = p_exact
    res['l1_p_err_pos'] = p_err_pos
    res['l1_p_exact_pos'] = p_exact_pos
    return res
