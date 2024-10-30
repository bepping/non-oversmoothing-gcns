import scipy
import numpy as np
from scipy.special import erf


class GCNGP():
    '''
    Calculate theoretical prediction of the GCNGP, such as covariance between features/preactivations,
    the equilibrium covariance or the condition for oversmoothing.
    '''
    def __init__(self, X, A, L=2, sigma_b_sq=0.1, sigma_w_sq=1.0):
        assert X.ndim == 2
        self.n_in = X.shape[1]
        if type(X) is np.ndarray:
            K_input = X @ X.T / self.n_in
        else:
            K_input = np.array((X @ X.T / self.n_in).todense())
        assert K_input.shape[0] == K_input.shape[1] and K_input.ndim == 2
        assert A.shape == K_input.shape
        # assert np.all(A == A.T)
        self.K_input = K_input
        self.X = X
        self.N = K_input.shape[0]
        self.A = A
        self.L = L
        self.sigma_b_sq = sigma_b_sq
        self.sigma_w_sq = sigma_w_sq
        self.all_Ks = np.zeros((self.N, self.N, L))
        self.all_Ks[:, :, 0] = self.sigma_b_sq + self.sigma_w_sq * self.A @ (self.A @ K_input).T
        self.masks = self._generate_masks(self.N)
        self._all_Ks_computed = False
        self.all_node_dist = None
        self.K_eq = None
        self.fully_correlated_K_eq = None
        self.linearized_perfect_corr = None

    def compute_linearization_perfect_corr(self):
        if self.fully_correlated_K_eq is None: self.compute_fully_correlated_K_eq()
        linearization_K = self.fully_correlated_K_eq * np.ones((self.N, self.N))
        self.linearized_perfect_corr = Linearized_GP(self.A, self.N, self.sigma_w_sq, linearization_K)

    def chaos_condition(self):
        assert self.linearized_perfect_corr is not None
        return np.max(abs(self.linearized_perfect_corr.eigvals))

    def compute_all_Ks(self):
        for l in range(self.L-1):
            self.all_Ks[:, :, l+1] = self._single_step(self.all_Ks[:, :, l])
        self._all_Ks_computed = True

    def compute_all_node_dist(self, return_Cs=False):
        if not self._all_Ks_computed: self.compute_all_Ks()
        all_Cs = np.zeros((self.N, self.N, self.L))
        for l in range(self.L):
            all_Cs[:, :, l] = self.C_from_K(self.all_Ks[:, :, l], self.masks)
        if return_Cs: return all_Cs
        self.all_node_dist = self._cov_dev_to_node_dist(all_Cs)

    def compute_K_eq(self, L_eq=2000, return_C=False):
        if not self._all_Ks_computed:
            self.compute_all_Ks()
        K = self.all_Ks[:, :, -1]
        for _ in range(L_eq - self.L):
            K = self.sigma_b_sq + self.sigma_w_sq * self.A @ (self.A @ self.C_from_K(K, self.masks)).T
        self.K_eq = K
        if return_C:
            return self.C_from_K(self.K_eq, self.masks)

    def C_from_K(self, K, masks):
        mask_diag, mask_offdiag, mask_diag_a, mask_diag_b = masks
        C = np.zeros_like(K)
        K_masked = K[np.tril_indices(self.N)]
        C_masked = np.zeros_like(K_masked)
        C_masked[mask_offdiag] = 2 / np.pi * np.arcsin(np.pi/2 * K_masked[mask_offdiag] /
                                                        np.sqrt(1 + np.pi/2 *K_masked[mask_diag_a]) /
                                                        np.sqrt(1 + np.pi/2 *K_masked[mask_diag_b]))
        C_masked[mask_diag] = 2 / np.pi * np.arcsin(K_masked[mask_diag] / (2/np.pi + K_masked[mask_diag]))
        C[np.tril_indices(self.N)] = C_masked
        C = C + C.T - np.diag(C_masked[mask_diag])
        return C

    def compute_fully_correlated_K_eq(self, L_eq=2000):
        assert np.allclose(np.sum(self.A, axis=0), 1.) and np.allclose(np.sum(self.A, axis=1), 1.)
        if self.fully_correlated_K_eq is None:
            K = 1.
            for _ in range(L_eq):
                K = self.sigma_w_sq * self._C_from_K_auto_scalar(K) + self.sigma_b_sq
            self.fully_correlated_K_eq = K

    def copy(self):
        return type(self)(self.X, self.A, self.L, self.sigma_b_sq, self.sigma_w_sq)

    def _C_from_K_auto_scalar(self, K):
        return 2 / np.pi * np.arcsin(K / (2/np.pi + K))

    def _single_step(self, K):
        return self.sigma_b_sq + self.sigma_w_sq * self.A @ (self.A @ self.C_from_K(K, self.masks)).T
    
    def _generate_masks(self, N):
        d = np.sum(range(N+1))
        # diagonal and off-diagonal masks
        mask_diag = np.cumsum(range(1, N+1)) - 1
        mask_offdiag = np.delete(range(d), mask_diag)
        # diagonal elements corresponding to the off-diagonal ones
        mask_diag_a = np.zeros_like(mask_offdiag)
        mask_diag_b = np.zeros_like(mask_offdiag)
        for i, x in enumerate(np.cumsum(range(0, N-1))):
            mask_diag_a[x:x+i+1] = mask_diag[:i+1]
            mask_diag_b[x:x+i+1] = mask_diag[i+1]
        return mask_diag, mask_offdiag, mask_diag_a, mask_diag_b
    
    def _cov_dev_to_node_dist(self, cov_dev):
        triu_inds_x, triu_inds_y = np.triu_indices(cov_dev.shape[0], 1)
        node_sim = np.zeros(cov_dev.shape[2])
        for l in range(cov_dev.shape[2]):
            for i in range(len(triu_inds_x)):
                node_sim[l] += (cov_dev[triu_inds_x[i], triu_inds_x[i], l]
                                + cov_dev[triu_inds_y[i], triu_inds_y[i], l]
                                - 2 * cov_dev[triu_inds_x[i], triu_inds_y[i], l])
        return node_sim / len(triu_inds_x)


class Linearized_GP():
    '''
    Calculate the linearization of a GCNGP around a the oversmoothed fixed point.
    '''
    def __init__(self, A, N, sigma_w_sq, linearization_K):
        self.N = N
        self.linearization_K = linearization_K
        self.eigvals = None
        self.unique_eigvals = None
        self.unique_eigvals_inds = None
        self.eigval_multiplicity = None
        self.eigvecs = None
        self.biorth_vecs = None
        self.num_of_eigenspaces = None
        self.transition_matrix = None
        self._compute_transition_matrix(A, N, sigma_w_sq)
        self._compute_eigvals()

    def _compute_transition_matrix(self, A, N, sigma_w_sq):
        assert np.allclose(np.sum(A, axis=0), 1.) and np.allclose(np.sum(A, axis=1), 1.)  # check double stochastic
        assert np.allclose(self.linearization_K, self.linearization_K[0, 0])  # check oversmoothed fixed point
        K_eq_scalar = self.linearization_K[0, 0]
        q = self._compute_q_erf_toy(K_eq_scalar)
        p_c = self._compute_p_c_erf_toy(K_eq_scalar, K_eq_scalar, K_eq_scalar)
        p_a = self._compute_p_a_erf_toy(K_eq_scalar, K_eq_scalar, K_eq_scalar)
        der = self._get_der_toy(N, q, p_c, p_a)
        der_A = np.einsum('ijkl,mj->imkl', der, A)
        A_der_A = np.einsum('ij,jklm->iklm', A, der_A).reshape(N**2, N**2)
        self.transition_matrix = sigma_w_sq * A_der_A

    def _compute_eigvals(self):
        self.eigvals = np.linalg.eigvals(self.transition_matrix)
        if np.allclose(self.eigvals.imag, 0): self.eigvals = self.eigvals.real
    
    def _get_der_toy(self, N, q, p_c, p_a):
        tensor = np.zeros((N, N, N, N), dtype=float)
        for i in range(N):
            tensor[i, i, i, i] += q
            for j in range(N-i-1):
                s = j + i + 1
                tensor[i, s, i, i] += p_a
                tensor[i, s, s, s] += p_a
                tensor[i, s, i, s] += 1/2 * p_c
                tensor[i, s, s, i] += 1/2 * p_c
                tensor[s, i, i, i] += p_a
                tensor[s, i, s, s] += p_a
                tensor[s, i, i, s] += 1/2 * p_c
                tensor[s, i, s, i] += 1/2 * p_c
        return tensor
    
    def _compute_q_erf_toy(self, K_eq_a):
        return 4 / np.pi**2 / np.sqrt(1 - (K_eq_a / (2 / np.pi + K_eq_a))**2) / (2/np.pi + K_eq_a)**2

    def _compute_p_c_erf_toy(self, Kc, Ka1, Ka2):
        return (2 / np.pi / np.sqrt(1 - Kc**2 / (2 / np.pi + Ka1) / (2 / np.pi + Ka2))
                / np.sqrt(2 / np.pi + Ka1) / np.sqrt(2 / np.pi + Ka2))

    def _compute_p_a_erf_toy(self, Ka1, Kc, Ka2):
        return (- 1 / np.pi / np.sqrt(1 - Kc**2 / (2 / np.pi + Ka1) / (2 / np.pi + Ka2))
                * Kc / np.sqrt(2 / np.pi + Ka1)**3 / np.sqrt(2 / np.pi + Ka2))


class GCN():
    '''
    Implementation for simulating untrained finite size GCNs to compare
    with the theoretical prediction of the GCNGP.
    '''
    def __init__(self, gcngp: GCNGP, n, rng=np.random.default_rng()):
        self.n = n
        self.Xs = np.zeros((gcngp.N, self.n, gcngp.L))
        self.Hs = np.zeros((gcngp.N, self.n, gcngp.L))
        self.all_Ks = np.zeros((gcngp.N, gcngp.N, gcngp.L))
        self.all_Cs = np.zeros((gcngp.N, gcngp.N, gcngp.L))
        self.gcngp = gcngp
        self.rng = rng

    def compute_Xs(self):
        W_in = self.rng.normal(0, np.sqrt(self.gcngp.sigma_w_sq / self.gcngp.n_in), (self.gcngp.n_in, self.n))
        b_in = self.rng.normal(0, np.sqrt(self.gcngp.sigma_b_sq), (self.n))
        self.Hs[:, :, 0] = self.gcngp.A @ self.gcngp.X @ W_in + b_in
        self.Xs[:, :, 0] = erf(np.sqrt(np.pi) * (self.Hs[:, :, 0]) / 2)
        for l in range(self.gcngp.L - 1):
            W = self.rng.normal(0, np.sqrt(self.gcngp.sigma_w_sq / self.n), (self.n, self.n))
            b = self.rng.normal(0, np.sqrt(self.gcngp.sigma_b_sq), (self.n))
            self.Hs[:, :, l+1] = self.gcngp.A @ self.Xs[:, :, l] @ W + b
            self.Xs[:, :, l+1] = erf(np.sqrt(np.pi) * (self.Hs[:, :, l+1]) / 2)

    def compute_all_Cs(self):
        self.all_Cs[:, :, :] = np.einsum('ilk,jlk->ijk', self.Xs, self.Xs) / self.n
        for l in range(self.gcngp.L):
            assert np.all(self.all_Cs[:, :, l] == self.all_Cs[:, :, l].T)


class complete_graph_GCNGP():
    '''
    Implementation to determine the chaos transition and the equilibrium of a GCNGP with
    error function non-linearity on a complete graph.
    '''
    def __init__(self, N, sigma_b_sq=0., sigma_w_sq=1., g=0.5):
        self.sigma_b_sq = sigma_b_sq
        self.sigma_w_sq = sigma_w_sq
        self.N = N
        self.g = g

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, value):
        self._g = value
        self.b = 1 - self.g
        self.eps = self.g / (self.N-1)
        self.g_a = (self.b**2 + (self.N-1) * self.eps**2) * self.sigma_w_sq
        self.g_c = 2 * ((self.N-1) * self.b * self.eps + (self.N-2) * (self.N-1) / 2 * self.eps**2) * self.sigma_w_sq
        self.h_a = (2 * self.b * self.eps + (self.N-2) * self.eps**2) * self.sigma_w_sq
        self.h_c = (self.b**2 + self.eps**2
                    + 2 * (self.N-2) * (self.N-3) / 2 * self.eps**2
                    + 2 * (self.N-2) * (self.eps * self.b + self.eps**2)) * self.sigma_w_sq

    def compute_K_eq(self, t_eq=2000, tol=1e-8):
        '''Calculate equilibrium covariance K and C for preactivations h and features x respectively.'''
        K_a_prev = 1
        K_c_prev = 0.5
        for t in range(t_eq):
            K_a_next, K_c_next = self._iteration_step(K_a_prev, K_c_prev)
            if abs(K_a_next-K_a_prev) < tol and abs(K_c_next-K_c_prev) < tol:
                return K_a_next, K_c_next, self._C_a(K_a_next), self._C_a(K_c_next)
            K_a_prev = K_a_next
            K_c_prev = K_c_next
        return K_a_next, K_c_next, self._C_a(K_a_next), self._C_a(K_c_next)

    def chaos_condition(self, g=None, for_minimize=False):
        if g is not None:
            self.g = g
        K_eq = self._compute_K_eq_perfect_corr()
        par_C_par_c = self._partial_C_partial_c_at_cis1(K_eq)
        chaos_cond = (self.h_c * par_C_par_c * K_eq - self.g_c * par_C_par_c * K_eq) / K_eq**2
        if for_minimize:
            return chaos_cond - 1.
        return chaos_cond

    def _C_a(self, K_a_prev):
        '''Calculate variance of features x from variance of preactivations h.'''
        return 2/np.pi * np.arcsin(np.pi / 2 * K_a_prev / (1 + np.pi / 2 * K_a_prev))

    def _C_c(self, K_c_prev, K_a_prev):
        '''Calculate covariance of features x from covariance of preactivations h.'''
        # for the error function and equal activity at each node
        return 2/np.pi * np.arcsin(np.pi / 2 * K_c_prev / (1 + np.pi / 2 * K_a_prev))

    def _iteration_step_perfect_corr(self, K_a_prev):
        '''One layer for the case of a network state with K_alpha,alpha=K_alpha,beta=K_beta,beta.'''
        return self.sigma_b_sq + (self.g_a + self.g_c) * self._C_a(K_a_prev)

    def _compute_K_eq_perfect_corr(self, t_eq=2000, tol=1e-8):
        # calculate equilibrium variance in perfect correlated state
        K_prev = 1
        for t in range(t_eq):
            K_next = self._iteration_step_perfect_corr(K_prev)
            if abs(K_next-K_prev) < tol:
                return K_next
            K_prev = K_next
        return K_next

    def _iteration_step(self, K_a_prev, K_c_prev):
        K_a_next = self.sigma_b_sq + self.g_a * self._C_a(K_a_prev) + self.g_c * self._C_c(K_c_prev, K_a_prev)
        K_c_next = self.sigma_b_sq + self.h_a * self._C_a(K_a_prev) + self.h_c * self._C_c(K_c_prev, K_a_prev)
        return K_a_next, K_c_next

    def _partial_C_partial_c_at_cis1(self, K_a):
        return 1 / np.sqrt(1 - (K_a / (2 / np.pi + K_a))**2) * K_a / (2 / np.pi + K_a) * 2 / np.pi


class CSBM():
    '''
    Generate adjacency and feature data as determined by the CSBM model.
    '''
    def __init__(self, N=10, gamma=1.5, lam=1, mu=1, d=3, rng=None):
        if rng is None: rng = np.random.default_rng()
        rng = rng or np.random.default_rng()
        assert N%2 == 0
        self.N = N
        self.gamma = gamma
        self.p = int(N / gamma)
        self.lam = lam
        self.mu = mu
        # irrelevant what u is (except from magnitude)
        # one can always transform into a system such that u = (1, 0, ..., 0)^T
        self.u = rng.normal(0, np.sqrt(1/self.p), self.p)
        # self.u = np.zeros(self.p)
        # self.u[0] = 1.
        self.d = d
        self.c_in = self.d + self.lam * np.sqrt(self.d)
        self.c_out = self.d - self.lam * np.sqrt(self.d)
        self.A = None
        self.B = None
        self.v = None
        self.rng = rng

    def generate_data(self):
        # adjacency matrix
        A = np.zeros((self.N, self.N))
        inds_upper = np.triu_indices(int(self.N/2), 1)
        inds_lower = (inds_upper[0] + int(self.N/2), inds_upper[1] + int(self.N/2))
        A[inds_upper] = (self.rng.random(len(inds_upper[0])) < self.c_in / self.N).astype(int)
        A[inds_lower] = (self.rng.random(len(inds_upper[0])) < self.c_in / self.N).astype(int)
        A[:int(self.N/2), int(self.N/2):] = (self.rng.random((int(self.N/2), int(self.N/2))) < self.c_out / self.N).astype(int)
        self.A = A + A.T
        # node features
        self.v = np.ones(self.N)
        self.v[int(self.N/2):] = -1
        self.B = (np.sqrt(self.mu / self.N) * self.v[:, np.newaxis] * self.u[np.newaxis, :]
                  + self.rng.normal(0, 1, (self.N, self.p)) / np.sqrt(self.p))


class GP_predictor():
    '''
    Predict unknown labels using the covariance matrix of test and training data given by a Gaussian process.
    '''
    def __init__(self, K, N, labels):
        assert K.shape == (N, N)
        assert labels.shape == (N,)
        self.labels = labels
        self.K = K
        self.N = N

    def predict(self, train_inds):
        test_inds = np.delete(np.arange(self.N), train_inds)
        K_DD = self.K[np.ix_(train_inds, train_inds)] + 1e-2 * np.eye(int(self.N/2))
        K_starD = self.K[np.ix_(train_inds, test_inds)]
        # use cholesky decomposition for numerical stability
        L = scipy.linalg.cholesky(K_DD, lower=True)
        alpha = scipy.linalg.cho_solve((L, True), self.labels[train_inds])
        pred_labels = np.dot(K_starD.T, alpha)

        gen_error = np.sum((pred_labels - self.labels[test_inds])**2) / len(test_inds)
        return gen_error
