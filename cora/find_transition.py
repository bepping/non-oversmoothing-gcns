import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from GCN_methods.models import GCNGP
from GCN_methods.utils import cov_to_dist
from GCN_methods.load_utils import load_data
from GCN_methods.plotting import plot_transition_from_numerics


def double_stochastic_A(adjacency_A, min_fraction_diag=0.6):
    assert min_fraction_diag > 0 and min_fraction_diag <= 1
    D = sp.csr_array(sp.diags(adjacency_A.sum(axis=0)))
    eps = (1 - min_fraction_diag) / D.max()
    A = sp.eye(adjacency_A.shape[0]) - eps * (D - adjacency_A)
    return A


def main():
    data = load_data('cora')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = data
    A = double_stochastic_A(adj, min_fraction_diag=0.1)

    # find transition
    num_sigma_w_sq = 31
    L_eq = 4000
    sigma_w_sq_arr = np.linspace(0.9, 1.2, num_sigma_w_sq)

    oversmoothing_measure = np.zeros_like(sigma_w_sq_arr)
    for i, sigma_w_sq in enumerate(tqdm(sigma_w_sq_arr)):
        gcngp = GCNGP(X=features, A=A, L=2, sigma_b_sq=0, sigma_w_sq=sigma_w_sq)
        C = gcngp.compute_K_eq(L_eq=L_eq, return_C=True)
        oversmoothing_measure[i] = np.max(cov_to_dist(C))

    plot_transition_from_numerics(sigma_w_sq_arr, oversmoothing_measure)


if __name__ == '__main__':
    main()
