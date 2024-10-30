import sklearn
import sklearn.linear_model
import numpy as np
from scipy.optimize import fsolve
from . import models


global_seed = 308455257035357856272718479280853966106


def A_complete_graph(N, g):
    inds = np.triu_indices(N, k=1)
    A = np.identity(N) * (1. - g)
    A[inds] = g / (N-1)
    A = A + A.T - np.diag(A.diagonal())
    return A


def double_stochastic_A(adjacency_A, g):
    D = np.diag(np.sum(adjacency_A, axis=0))
    return np.eye(adjacency_A.shape[0]) - g / np.max(D) * (D - adjacency_A)


def cov_to_dist(K):
    dist = np.zeros_like(K)
    assert K.ndim == 2 and K.shape[0] == K.shape[1]
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            dist[i, j] = K[i, i] + K[j, j] - 2 * K[i, j]
    return dist


def avg_dists_blockwise(dist_arr):
    _, N, L = dist_arr.shape
    assert np.isclose(int(N/2), N/2)
    inds_x, inds_y = np.triu_indices(int(N/2), 1)

    diag_1 = dist_arr[0:int(N/2), 0:int(N/2), :]
    diag_2 = dist_arr[int(N/2):, int(N/2):, :]
    offdiag = dist_arr[0:int(N/2), int(N/2):, :]
    diag_1_mean = np.mean(diag_1[inds_x, inds_y, :], axis=0)
    diag_2_mean = np.mean(diag_2[inds_x, inds_y, :], axis=0)
    offdiag_mean = np.mean(offdiag, axis=(0,1))
    return (diag_1_mean + diag_2_mean)/2, offdiag_mean


def node_distance_from_sim(Hs):
    measure = np.zeros(Hs.shape[2])
    triu_inds_x, triu_inds_y = np.triu_indices(Hs.shape[0], 1)
    for l in range(len(measure)):
        for i in range(len(triu_inds_x)):
            measure[l] += (np.linalg.norm(Hs[triu_inds_x[i], :, l] - Hs[triu_inds_y[i], :, l])**2
                           / Hs.shape[1])
    return measure / len(triu_inds_x)


def chaos_cond_from_sigma_w_sq_minus_one(sigma_w_sq, gcngp: models.GCNGP):
    gcngp_copy = models.GCNGP(gcngp.X, gcngp.A, sigma_w_sq=gcngp.sigma_w_sq, sigma_b_sq=gcngp.sigma_b_sq)
    gcngp_copy.sigma_w_sq = sigma_w_sq
    gcngp_copy.compute_fully_correlated_K_eq()
    gcngp_copy.compute_linearization_perfect_corr()
    return gcngp_copy.chaos_condition() - 1.


def get_critical_sigma_w_sq(gcngp: models.GCNGP):
    sigma_w_sq_in_arr, infodict, success, _ = fsolve(func=chaos_cond_from_sigma_w_sq_minus_one,
                                              x0=3, args=(gcngp,), full_output=True)
    assert success == 1
    return sigma_w_sq_in_arr[0], infodict['nfev']


def eval_performance_mult_L(csbm: models.CSBM, sigma_w_sq_arr, A, L_arr, sigma_b_sq):
    gen_error_arr = np.zeros((len(sigma_w_sq_arr), len(L_arr)))
    train_inds = np.append(np.arange(int(csbm.N/4)), np.arange(int(csbm.N/4)) + int(csbm.N/2))
    for i, sigma_w_sq in enumerate(sigma_w_sq_arr):
        gcngp = models.GCNGP(csbm.B, A, L=np.max(L_arr)+1, sigma_b_sq=sigma_b_sq,
                        sigma_w_sq=sigma_w_sq)
        gcngp.compute_all_Ks()
        for l, L in enumerate(L_arr):
            # regarding inidexing of gcngp.all_Ks:
            # in gcngp.all_Ks[:, :, l] is the covariance of linear readouts of an l hidden layer GCN
            gp_predictor = models.GP_predictor(gcngp.all_Ks[:, :, L], csbm.N, csbm.v)
            gen_error_arr[i, l] = gp_predictor.predict(train_inds)
    return gen_error_arr


def eval_performace_finite_size_mult_L(csbm: models.CSBM, gcngp: models.GCNGP, n, L_arr, rng,
                                       train_inds, test_inds, sigma_ro=0.01):
        y_train = csbm.v[train_inds]
        y_test = csbm.v[test_inds]
        gcn = models.GCN(gcngp, n, rng)
        gcn.compute_Xs()
        gen_error_arr = np.zeros_like(L_arr, dtype=float)

        # the -1 necassary since readout layer is added explicitly and thus needs to be subtracted from the GCN
        for i, l in enumerate(L_arr-1):
            X_train = gcn.Xs[train_inds, :, l] + rng.normal(0, sigma_ro, gcn.Xs[train_inds, :, l].shape)
            X_test = gcn.Xs[test_inds, :, l] + rng.normal(0, sigma_ro, gcn.Xs[test_inds, :, l].shape)

            lin_class = sklearn.linear_model.SGDRegressor(random_state=rng.integers(0, 4294967295))
            lin_class.fit(X_train, y_train)
            y_predict = lin_class.predict(X_test)
            gen_error_arr[i] = np.sum((y_predict - y_test)**2) / len(y_test)
        return gen_error_arr
