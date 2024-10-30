import numpy as np
from tqdm import tqdm
from scipy.optimize import fsolve
from GCN_methods.models import complete_graph_GCNGP
from GCN_methods.plotting import plot_distance_phase_diagram


def main():
    # Test chaos transition prediction on a complete graph.
    N = 5
    sigma_w_sq_arr = np.linspace(0.7, 8, 100)
    g_arr = np.linspace(0., 0.2, 100)

    erf_gcngp = complete_graph_GCNGP(N)
    dist_arr = np.zeros((len(sigma_w_sq_arr), len(g_arr)))
    transition_pred = np.zeros((len(sigma_w_sq_arr)))
    for j, sigma_w_sq in enumerate(tqdm(sigma_w_sq_arr)):
        for k, g in enumerate(g_arr):
            erf_gcngp.sigma_w_sq = sigma_w_sq
            erf_gcngp.g = g
            # Determine the equilibrium distances by iterating the GCNGP
            _, _, C_a_eq, C_c_eq = erf_gcngp.compute_K_eq(t_eq=4000)
            dist_arr[j, k] = 2* C_a_eq - 2* C_c_eq
            # Determine the critical value for g using the theoretical prediction
            g_sol, _, success, _ = fsolve(func=erf_gcngp.chaos_condition, x0=0, args=(True,), full_output=True)
            assert success == 1
            transition_pred[j] = g_sol[0]

    plot_distance_phase_diagram(sigma_w_sq_arr, g_arr, dist_arr, transition_pred)


if __name__ == '__main__':
    main()
