import numpy as np
import networkx as nx
from tqdm import tqdm
from GCN_methods.models import CSBM, GCNGP
from GCN_methods.plotting import plot_KW_oversmoothing
from GCN_methods.utils import global_seed, double_stochastic_A


def kipf_welling_A(adj_A):
    A_tilde = adj_A + np.eye(adj_A.shape[0])
    D_sqinv = np.diag(1/np.sqrt(np.sum(A_tilde, axis=0)))
    return D_sqinv @ A_tilde @ D_sqinv


def cov_to_node_sim(cov):
    triu_inds_x, triu_inds_y = np.triu_indices(cov.shape[0], 1)
    node_sim = 0
    for i in range(len(triu_inds_x)):
        node_sim += (cov[triu_inds_x[i], triu_inds_x[i]]
                     + cov[triu_inds_y[i], triu_inds_y[i]]
                     - 2 * cov[triu_inds_x[i], triu_inds_y[i]])
    return node_sim / len(triu_inds_x)


def main():
    mother_ssq = np.random.SeedSequence(global_seed)
    default_rng = np.random.default_rng(np.random.SeedSequence((global_seed, 4444)))
    N = 30
    n_csbms = 5
    # graph parameter
    lam = 1.
    d = 5
    # covariat parameter
    gamma = 1.
    mu = 4.
    # GP parameter
    L_eq = 1000
    # experiment
    num_sigs = 20
    sigma_w_sq_arr = np.linspace(0.5, 1.7, num_sigs)
    assert lam <= np.sqrt(d)
    # generate CSBM
    X = default_rng.normal(0, 1, (N, 5))

    # ending kw for the shift operator from Kipf and Welling
    node_sim_kw = np.zeros((num_sigs, n_csbms))
    zero_state_kw = np.zeros((num_sigs, n_csbms))
    max_K_kw = np.zeros((num_sigs, n_csbms))
    # ending ds for the double stochastic shift operator
    node_sim_ds = np.zeros((num_sigs, n_csbms))
    zero_state_ds = np.zeros((num_sigs, n_csbms))
    max_K_ds = np.zeros((num_sigs, n_csbms))

    ssq_list = mother_ssq.spawn(n_csbms)
    for j, ssq in enumerate(tqdm(ssq_list)):
        rng = np.random.default_rng(ssq)
        connected = False
        while not connected:
            csbm = CSBM(N, gamma, lam, mu, d, rng)
            csbm.generate_data()
            G = nx.from_numpy_array(csbm.A)
            connected = nx.is_connected(G)
        assert np.all(np.sum(csbm.A, axis=0) > 0)
        # calculate the GCNGP
        A_kw = kipf_welling_A(csbm.A)
        A_ds = double_stochastic_A(csbm.A, g=0.3)

        for i, sigma_w_sq in enumerate(sigma_w_sq_arr):
            # calculate the GCNGP for the kw shift operator
            gcngp = GCNGP(X, A_kw, sigma_w_sq=sigma_w_sq, sigma_b_sq=0)
            gcngp.compute_K_eq(L_eq=L_eq)
            node_sim_kw[i, j] = cov_to_node_sim(gcngp.K_eq)
            zero_state_kw[i, j] = np.allclose(gcngp.K_eq, 0, atol=1e-5)
            max_K_kw[i, j] = np.max(gcngp.K_eq)
            # calculate the GCNGP for the ds shift operator
            gcngp = GCNGP(X, A_ds, sigma_w_sq=sigma_w_sq, sigma_b_sq=0)
            gcngp.compute_K_eq(L_eq=L_eq)
            node_sim_ds[i, j] = cov_to_node_sim(gcngp.K_eq)
            zero_state_ds[i, j] = np.allclose(gcngp.K_eq, 0, atol=1e-5)
            max_K_ds[i, j] = np.max(gcngp.K_eq)

    av_node_sim_kw = np.mean(node_sim_kw, axis=1)
    std_node_sim_kw = np.std(node_sim_kw, axis=1)
    av_node_sim_ds = np.mean(node_sim_ds, axis=1)
    std_node_sim_ds = np.std(node_sim_ds, axis=1)

    plot_KW_oversmoothing(sigma_w_sq_arr, av_node_sim_kw, std_node_sim_kw, max_K_kw, node_sim_kw,
                          av_node_sim_ds, std_node_sim_ds, max_K_ds, node_sim_ds)


if __name__ == '__main__':
    main()
