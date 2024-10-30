import numpy as np
import networkx as nx
from tqdm import tqdm
from GCN_methods.models import CSBM, GCNGP, GCN
from GCN_methods.plotting import plot_finite_size_gen_error
from GCN_methods.utils import (double_stochastic_A, get_critical_sigma_w_sq,
                               eval_performace_finite_size_mult_L, global_seed)


def main():
    mother_ssq = np.random.SeedSequence(global_seed)
    default_rng = np.random.default_rng(np.random.SeedSequence((global_seed, 4444)))
    N = 20
    n_csbms = 20
    # graph parameter
    lam = 1.
    d = 5
    # covariat parameter
    gamma = 1.
    mu = 4.
    # GP parameter
    L = 1000
    sigma_b_sq = 0.
    g = 0.1
    # experiment
    L_arr = np.unique(np.logspace(0, np.log10(L), 100, endpoint=True).astype(int))
    n = 200
    sigma_w_sq_dist = 1.
    assert int(N/4) == N/4  # for split to training/test data
    train_inds = np.append(np.arange(int(N/4)), np.arange(int(N/4)) + int(N/2))
    test_inds = np.delete(np.arange(N), train_inds)
    assert lam <= np.sqrt(d)

    gen_error_chaos = np.zeros((len(L_arr), n_csbms))
    gen_error_oversmoothing = np.zeros_like(gen_error_chaos)
    gen_error_crit = np.zeros_like(gen_error_chaos)

    ssq_list = mother_ssq.spawn(n_csbms)
    for i, ssq in enumerate(tqdm(ssq_list)):
        rng = np.random.default_rng(ssq)
        connected = False
        while not connected:
            csbm = CSBM(N, gamma, lam, mu, d, rng)
            csbm.generate_data()
            G = nx.from_numpy_array(csbm.A)
            connected = nx.is_connected(G)
        assert np.all(np.sum(csbm.A, axis=0) > 0)
        # calculate the GCNGP
        A_ds = double_stochastic_A(csbm.A, g)
        gcngp = GCNGP(csbm.B, A_ds, sigma_w_sq=1., sigma_b_sq=sigma_b_sq)
        crit_sigma_w_sq, _ = get_critical_sigma_w_sq(gcngp)

        # chaos performance
        gcngp = GCNGP(csbm.B, A_ds, L=L, sigma_w_sq=crit_sigma_w_sq + sigma_w_sq_dist, sigma_b_sq=sigma_b_sq)
        gen_error_chaos[:, i] = eval_performace_finite_size_mult_L(csbm, gcngp, n, L_arr, rng, train_inds, test_inds)

        # oversmoothing performance
        assert crit_sigma_w_sq - sigma_w_sq_dist > 0
        gcngp = GCNGP(csbm.B, A_ds, L=L, sigma_w_sq=crit_sigma_w_sq - sigma_w_sq_dist, sigma_b_sq=sigma_b_sq)
        gen_error_oversmoothing[:, i] = eval_performace_finite_size_mult_L(csbm, gcngp, n, L_arr, rng, train_inds, test_inds)

        # at critical line
        gcngp = GCNGP(csbm.B, A_ds, L=L, sigma_w_sq=crit_sigma_w_sq, sigma_b_sq=sigma_b_sq)
        gen_error_crit[:, i] = eval_performace_finite_size_mult_L(csbm, gcngp, n, L_arr, rng, train_inds, test_inds)

    plot_finite_size_gen_error(L_arr, np.mean(gen_error_chaos, axis=1),
                               np.mean(gen_error_oversmoothing, axis=1),
                               np.mean(gen_error_crit, axis=1))


if __name__ == '__main__':
    main()
