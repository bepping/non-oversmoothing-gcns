import numpy as np
import scipy as sc
from tqdm import tqdm
import scipy.sparse as sp
from GCN_methods.models import GCNGP
from GCN_methods.load_utils import load_data
from GCN_methods.plotting import plot_gen_error_hm


class GP_predictor_cora():
    # different implementation due to different storage of labels compared to CSBM
    def __init__(self, K, N):
        assert K.shape == (N, N)
        self.K = K
        self.N = N

    def predict(self, train_inds, train_labels, test_inds, test_labels):
        K_DD = self.K[np.ix_(train_inds, train_inds)] + 1e-2 * np.eye(len(train_inds))
        K_starD = self.K[np.ix_(train_inds, test_inds)]
        L = sc.linalg.cholesky(K_DD, lower=True)
        alpha = sc.linalg.cho_solve((L, True), train_labels)
        pred_labels = np.dot(K_starD.T, alpha)

        pred_classes = np.argmax(pred_labels, axis=1)
        test_classes = np.argmax(test_labels, axis=1)
        accuracy = np.sum(test_classes == pred_classes) / len(pred_classes)
        gen_error = np.sum((pred_labels - test_labels)**2) / len(test_inds)
        return accuracy, gen_error


def double_stochastic_A(adjacency_A, g):
    # scipy sparse implementation, therefore we do not use the version from GCN_methods.utils
    min_fraction_diag = 1 - g
    assert min_fraction_diag > 0 and min_fraction_diag <= 1
    D = sp.csr_array(sp.diags(adjacency_A.sum(axis=0)))
    eps = (1 - min_fraction_diag) / D.max()
    A = sp.eye(adjacency_A.shape[0]) - eps * (D - adjacency_A)
    return A


def compute_covariances(A, features, sigma_w_sq=1., L=3):
    gcngp = GCNGP(X=features, A=A, L=L, sigma_b_sq=0, sigma_w_sq=sigma_w_sq)
    gcngp.compute_all_Ks()
    return gcngp.all_Ks


def main():
    data = load_data('cora')
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = data

    A = double_stochastic_A(adj, g=0.9)

    train_inds = np.where(train_mask)[0]
    test_inds = np.where(test_mask)[0]

    train_labels = y_train[train_inds, :]
    test_labels = y_test[test_inds, :]

    # multiple sigma_w_sq and L
    num_sigma_w_sq = 25
    L = 100
    gen_err_arr = np.zeros((num_sigma_w_sq, L), dtype=float)
    acc_arr = np.zeros((num_sigma_w_sq, L), dtype=float)
    sigma_w_sq_arr = np.linspace(0.5, 2.5, num_sigma_w_sq)
    for i, sigma_w_sq in enumerate(tqdm(sigma_w_sq_arr)):
        Ks = compute_covariances(A, features, L=L, sigma_w_sq=sigma_w_sq)
        for j in range(L):
            predictor = GP_predictor_cora(Ks[:, :, j], A.shape[0])
            acc_arr[i, j], gen_err_arr[i, j] = predictor.predict(train_inds, train_labels, test_inds, test_labels)

    plot_gen_error_hm(gen_err_arr, sigma_w_sq_arr, np.arange(L))
    plot_gen_error_hm(acc_arr, sigma_w_sq_arr, np.arange(L), label='accuracy')


if __name__ == '__main__':
    main()
