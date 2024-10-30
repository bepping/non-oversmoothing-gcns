import numpy as np
from tqdm import tqdm
from GCN_methods.models import GCNGP, CSBM
from GCN_methods.utils import double_stochastic_A, cov_to_dist, global_seed
from GCN_methods.plotting import plot_transition_pred_test, plot_distance_matrix


def main():
    default_rng = np.random.default_rng(np.random.SeedSequence((global_seed, 4444)))
    N = 8
    # CSBM graph parameter
    lam = 1.5
    d = 5
    # CSBM covariat parameter (irrelevant here, only interested in equilibrium)
    gamma = 1.
    mu = 4.
    # GP parameter
    sigma_b_sq = 0.
    g = 0.3
    # GCNGP parameter
    sigma_w_sq_arr = np.linspace(1.1, 2.1, 31)
    dist_eq_inds = np.array([0, len(sigma_w_sq_arr)-1])  # save distance matrices at these points
    dist_eq_arr = np.zeros((N, N, len(dist_eq_inds)))
    assert lam <= np.sqrt(d)
    # generate CSBM
    csbm = CSBM(N, gamma, lam, mu, d, default_rng)
    csbm.generate_data()

    X = default_rng.normal(0, 1, (N, 5))
    A = double_stochastic_A(csbm.A, g)
    max_dist = np.zeros(len(sigma_w_sq_arr))
    chaos_cond = np.zeros_like(max_dist)

    t = 0
    for i, sigma_w_sq in enumerate(tqdm(sigma_w_sq_arr)):
        gcngp = GCNGP(X, A, sigma_w_sq=sigma_w_sq, sigma_b_sq=sigma_b_sq)
        # determine equilibrium of GCNGP
        gcngp.compute_K_eq(L_eq=4000)
        gcngp.compute_fully_correlated_K_eq()
        masks = gcngp._generate_masks(gcngp.N)
        C_eq = gcngp.C_from_K(gcngp.K_eq, masks)
        dist = cov_to_dist(C_eq)
        max_dist[i] = np.max(dist)
        # calculate chaos transition
        gcngp.compute_linearization_perfect_corr()
        chaos_cond[i] = gcngp.chaos_condition()
        # save distance matrices at given points
        if i in dist_eq_inds: 
            dist_eq_arr[:, :, t] = dist
            t += 1

    plot_transition_pred_test(sigma_w_sq_arr, max_dist, chaos_cond)
    plot_distance_matrix(dist_eq_arr[:, :, -1])


if __name__ == '__main__':
    main()
