import numpy as np
from tqdm import tqdm
from GCN_methods.models import GCNGP, GCN
from GCN_methods.plotting import plot_node_distance_over_layers
from GCN_methods.utils import A_complete_graph, node_distance_from_sim, global_seed


def main():
    # Test quantitative prediction for node distances.
    mother_ssq = np.random.SeedSequence(global_seed)
    default_rng = np.random.default_rng(np.random.SeedSequence((global_seed, 4444)))
    L = 1000
    sigma_b_sq = 0.
    N = 5
    sigma_w_sq = 2.
    g = 0.18
    n_in = 20  # input dimension
    n = 200  # dimension of feature space on nodes for simulation
    n_networks = 50

    # Generate random input data on the graph
    X = default_rng.normal(0, 1, (N, n_in))

    # Generate shift operator
    A = A_complete_graph(N, g)
    # Evaluate the GCNGP
    gcngp = GCNGP(X, A, L, sigma_b_sq, sigma_w_sq)
    gcngp.compute_all_Ks()
    gcngp.compute_all_node_dist()
    # Simulate finite size networks
    sim_all_node_similarities = np.zeros((L, n_networks))
    ssq_list = mother_ssq.spawn(n_networks)
    for i, ssq in enumerate(tqdm(ssq_list)):
        rng = np.random.default_rng(ssq)
        gcn = GCN(gcngp, n, rng)
        gcn.compute_Xs()
        sim_all_node_similarities[:, i] = node_distance_from_sim(gcn.Xs)

    plot_node_distance_over_layers(L, gcngp.all_node_dist, np.mean(sim_all_node_similarities, axis=1))


if __name__ == '__main__':
    main()
