
"""Helper functions"""
from __future__ import division
from gurobipy import *
import numpy as np


def prepareSparseTensor(input_csrMat):
    """Extract the indices/values/shapes from a csr_matrix"""
    batch_inputs = input_csrMat.tocoo()
    batch_indices = [batch_inputs.row, batch_inputs.col]
    batch_indices = np.vstack(batch_indices).T.astype(np.int64)
    batch_values = batch_inputs.data.astype(np.float32)
    return batch_indices, batch_values, np.array(input_csrMat.shape)


def lp_bp_err(A, y, true_x):
    """
    Solve min_x ||x||_1 s.t. Ax=y, and compute err = ||x-true_x||_2.
    To convert it to the form of an LP:
    min_i \sum_i h_i s.t. -h_i <= x_i <= h_i, Ax=y
    """
    emb_dim, input_dim = A.shape
    model = Model()
    model.params.outputflag = 0   # disable solver output
    x = []
    for i in range(input_dim):
        x.append(model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=0))
    for i in range(input_dim):
        x.append(model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, obj=1))
    model.update()
    # add inequality constraints
    for i in range(input_dim):
        model.addConstr(x[i] - x[i+input_dim] <= 0)
        model.addConstr(x[i] + x[i+input_dim] >= 0)
    # add equality constraints
    for i in range(emb_dim):
        coeff = A[i, :]
        expr = LinExpr(coeff, x[:input_dim])
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=y[i])
    # optimize the models and obtain the results
    model.optimize()
    res = []
    for v in model.getVars():
        res.append(v.x)
    temp_err = np.linalg.norm(res[:input_dim]-true_x)
    return temp_err, res[:input_dim]


def lp_bp_pos(A, y, true_x):
    """
    Solve min_x sum_ix_i s.t. Ax=y, x_i>= 0 and compute err = ||x-true_x||_2
    """
    emb_dim, input_dim = A.shape
    model = Model()
    model.params.outputflag = 0  # disable solver output # stop the printouts
    x = []
    for i in range(input_dim):
        # The lower bound lb=0.0 indicates that x>=0
        x.append(model.addVar(lb=0.0, ub=GRB.INFINITY, obj=1))
    model.update()
    # add equality constraints
    for i in range(emb_dim):
        coeff = A[i, :]
        expr = LinExpr(coeff, x)
        model.addConstr(lhs=expr, sense=GRB.EQUAL, rhs=y[i])
    # optimize the models and obtain the results
    model.optimize()
    res = []
    for v in model.getVars():
        res.append(v.x)
    temp_err = np.linalg.norm(res[:input_dim]-true_x)
    return temp_err, res[:input_dim]


def LP_BP_avg_err(A, Y, true_X, use_pos=False, eps=1e-8):
    """
    Run l1_min for each sample, and compute the RMSE.
    true_X is a 2D csr_matrix with shape=(num_sample, input_dim).
    """
    #global Res
    num_sample = Y.shape[0]
    num_exact = 0  # number of samples that are exactly recovered
    num_solved = num_sample  # number of samples that successfully runs l1_min
    err = 0
    for i in range(num_sample):
        y = Y[i, :].reshape(-1,)
        x = true_X[i, :].toarray().reshape(-1,)
        try:
            if use_pos:
                temp_err, Res = lp_bp_pos(A, y, x)
            else:
                temp_err, Res = lp_bp_err(A, y, x)
            if temp_err < eps:
                num_exact += 1
            err += temp_err**2  # squared error
        except Exception:
            num_solved -= 1
    avg_err = np.sqrt(err/num_solved)  # RMSE
    exact_ratio = num_exact/float(num_sample)
    solved_ratio = num_solved/float(num_sample)
    return avg_err, exact_ratio, solved_ratio