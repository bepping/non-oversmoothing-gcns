import numpy as np
from tqdm import tqdm
from GCN_methods.models import CSBM, GCNGP, GCN
from GCN_methods.plotting import plot_dist_block_avg
from GCN_methods.utils import double_stochastic_A, cov_to_dist, avg_dists_blockwise, global_seed


def main():
    mother_ssq = np.random.SeedSequence(global_seed)
    default_rng = np.random.default_rng(np.random.SeedSequence((global_seed, 4444)))
    N = 8
    L = 1000
    # graph parameter
    lam = 1.5
    d = 5
    # covariat parameter (irrelevant for this experiment)
    gamma = 1.
    mu = 4.
    # GCNGP parameter
    sigma_b_sq = 0.
    g = 0.3
    # simulation parameter
    n = 200
    n_networks = 20
    sigma_w_sq = 2.
    assert lam <= np.sqrt(d)
    # generate CSBM
    csbm = CSBM(N, gamma, lam, mu, d, default_rng)
    csbm.generate_data()

    X = default_rng.normal(0, 1, (N, 5))
    A = double_stochastic_A(csbm.A, g)

    # theory prediction
    gcngp = GCNGP(X, A, L, sigma_b_sq, sigma_w_sq)
    gcngp.compute_all_Ks()
    gcngp_all_Cs = gcngp.compute_all_node_dist(return_Cs=True)
    theo_dist_arr = np.zeros_like(gcngp_all_Cs)
    for i in range(L):
        theo_dist_arr[:, :, i] = cov_to_dist(gcngp_all_Cs[:, :, i])
    pred_diag, pred_offdiag = avg_dists_blockwise(theo_dist_arr)

    # Simulate finite size networks
    gcn_dist_arr = np.zeros_like(gcngp_all_Cs)
    ssq_list = mother_ssq.spawn(n_networks)
    for i, ssq  in enumerate(tqdm(ssq_list)):
        rng = np.random.default_rng(ssq)
        gcn = GCN(gcngp, n, rng)
        gcn.compute_Xs()
        gcn.compute_all_Cs()
        for i in range(L):
            gcn_dist_arr[:, :, i] = cov_to_dist(gcngp_all_Cs[:, :, i])
        sim_diag, sim_offdiag = avg_dists_blockwise(gcn_dist_arr)

    plot_dist_block_avg(L, pred_diag, sim_diag, pred_offdiag, sim_offdiag)


if __name__ == '__main__':
    main()
