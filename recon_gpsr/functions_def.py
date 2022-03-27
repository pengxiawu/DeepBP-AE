"""
Define various functions for algorithm implement

DC_GPSR_2loopBBmono: monotone DC_GPSR with BB step sizes, the inner loop is grad_proj_descent_BBmono
grad_proj_descent_BBmono: monotone GPSR with BB step sizes

"""

import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize.linesearch import line_search_armijo

"""
    plot setup
"""
kwargs = {'linewidth' : 3.5}
font = {'weight' : 'normal', 'size'   : 24}
matplotlib.rc('font', **font)

def error_plot(ys, yscale='log'):
    plt.figure(figsize=(5, 5))
    plt.xlabel('Step')
    plt.ylabel('Error')
    plt.yscale(yscale)
    plt.plot(range(len(ys)), ys, **kwargs)


"""
    gradient projection sparse reconstruction
    GPSR-basic, GPSR-linesearch and GPSR-BB
"""

def grad_proj_descent_BBmono(init, steps, grad, proj=lambda x: x):
    """
    Monotone gradient projection descent with BB step sizes

    Inputs:
        initial: starting point
        steps: list of scalar step sizes
        grad: function mapping points to gradients
        obj : function of objective
        proj (optional): function mapping points to points

    Returns:
        List of all points computed by projected gradient descent.
        List of used step sizes
    """
    xs = [init]
    step_BB = 1.0  # initialize step size
    for i in range(np.size(steps)):
        xs_update = proj(xs[-1] - step_BB * grad(xs[-1]))
        delta_xs = xs_update - xs[-1]
        delta_grad = grad(xs_update) - grad(xs[-1])
        if np.dot(delta_xs.T, delta_grad) == 0:
            step_BB = steps[i]
            step_lambda = 1.0
        else:
            step_BB = np.dot(delta_xs.T, delta_xs) / np.dot(delta_xs.T, delta_grad)
            steps[i] = step_BB
            step_lambda = - np.dot(delta_xs.T, grad(xs[-1])) / np.dot(delta_xs.T, delta_grad)
        step_lambda = min(1.0, step_lambda)
        step_lambda = max(0, step_lambda)
        xs_update_mono = xs[-1] + step_lambda * delta_xs
        xs.append(xs_update_mono)
        if np.linalg.norm(xs[-1] - xs[-2]) < 1e-30:
            break
        else:
            continue
    return xs, steps



"""
    Double-loop DC programming gradient projection sparse reconstruction (double-loop DC_GPSR) 
"""
def DC_GPSR_2loopBBmono(init, inner_steps_list, grad, proj, outer_loop_num, sparsity, penalty):
    """
    DC projected gradient descent with monotone BB step sized innner GPSR
    Iuputs:
        initial: starting point
        steps: list of scalar step sizes
        sparsity: an estimate of the number of nonzero elements
    """
    xs = [init]
    xs_outer_loop = [init]
    idx = [0]
    # time_idx = [0]
    outer_step = 0
    while outer_step < outer_loop_num:
        # start_time = time.time()
        s = K_norm_grad(xs[-1], sparsity)
        dc_grad = lambda x: grad(x) - penalty*s
        xs += grad_proj_descent_BBmono(xs[-1], inner_steps_list, dc_grad, proj)[0]
        # time_idx.append(time_idx[-1]+(time.time()-start_time))
        idx.append(len(xs)-1)
        xs_outer_loop.append(xs[-1])
        outer_step += 1
    return xs, xs_outer_loop, idx


"""
    objective functions
"""
def least_squares(A, b, m, x):
    """Least squares objective."""
    return (0.5/m) * np.linalg.norm(A.dot(x)-b)**2

"""
    gradient calculations
"""
def least_squares_gradient(A, b, m, x):
    """Gradient of least squares objective at x."""
    return A.T.dot(A.dot(x)-b)/m

def ell1_subgradient(x):
    """Subgradient of the ell1-norm at x."""
    g = np.ones(x.shape)
    g[x < 0.] = -1.0
    return g

def lasso_subgradient(A, b, m, x, alpha=0.1):
    """Subgradient of the lasso objective at x"""
    return least_squares_gradient(A, b, m, x) + alpha * ell1_subgradient(x)

def lasso_gradient(A, b, m, x, alpha=0.1):
    """gradient of lasso objective at x"""
    return least_squares_gradient(A, b, m, x) + alpha

def K_norm_grad(x, sparsity):
    temp = np.zeros(x.shape)
    temp[:] = x
    sort_temp = np.argsort(temp)
    ind_s = sort_temp[:len(temp)-sparsity]
    ind_l = sort_temp[len(temp)-sparsity:]
    temp[ind_s] = 0
    temp[ind_l] = 1
    return temp

"""
    projection onto nonnegative orthant
"""
def proj_nonnegative(x):
    return np.maximum(x, np.zeros(x.shape))

