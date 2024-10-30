import numpy as np
import networkx as nx
from tqdm import tqdm
from GCN_methods.models import CSBM, GCNGP
from GCN_methods.plotting import plot_gen_error_hm, plot_hist_crit_sigma_w_sq
from GCN_methods.utils import double_stochastic_A, get_critical_sigma_w_sq, eval_performance_mult_L, global_seed


def main():
    mother_ssq = np.random.SeedSequence(global_seed)
    default_rng = np.random.default_rng(np.random.SeedSequence((global_seed, 4444)))
    N = 20
    n_csbms = 30
    # graph parameter
    lam = 1.
    d = 5
    # covariat parameter
    gamma = 1.
    mu = 4.
    # GP parameter
    L_arr = np.arange(50)
    sigma_b_sq = 0.
    g = 0.1
    # experiment
    sigma_w_sq_arr = np.linspace(0.1, 7, 40)
    assert lam <= np.sqrt(d)
    assert int(N/4) == N/4 # for split to training/test data
    # generate CSBM
    X = default_rng.normal(0, 1, (N, 5))  # only used to determine chaos condition

    ssq_list = mother_ssq.spawn(n_csbms)
    crit_sigma_w_sq_arr = np.zeros(n_csbms)
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
        gcngp = GCNGP(X, A_ds, sigma_w_sq=1., sigma_b_sq=0)
        crit_sigma_w_sq_arr[i], _ = get_critical_sigma_w_sq(gcngp)

        gen_error = eval_performance_mult_L(csbm, sigma_w_sq_arr, A_ds, L_arr, sigma_b_sq)

    plot_gen_error_hm(gen_error, sigma_w_sq_arr, L_arr)
    plot_hist_crit_sigma_w_sq(crit_sigma_w_sq_arr)


if __name__ == '__main__':
    main()
