import matplotlib.pyplot as plt
import numpy as np


def plot_distance_phase_diagram(sigma_w_sq_arr, eps_arr, dist_arr, transition_pred):
    fig, ax = plt.subplots(figsize=(4, 3))
    # simulation
    x, y = np.meshgrid(sigma_w_sq_arr, eps_arr)
    c = ax.pcolormesh(x, y, dist_arr.T, cmap='Blues_r')
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    ax.set_xlabel(r'$\sigma_w^2$')
    ax.set_ylabel(r'$g$')
    fig.colorbar(c, ax=ax)
    # theory
    ax.plot(sigma_w_sq_arr, transition_pred, color='red')
    plt.tight_layout()
    plt.show()


def plot_node_distance_over_layers(L, node_dist_theo, node_dist_sim):
    _, ax = plt.subplots(figsize=(4, 3))
    ax.plot(np.arange(L)+1, node_dist_sim, linestyle='', marker='x', label='simulation')
    ax.plot(np.arange(L)+1, node_dist_theo, label='theory')
    ax.set_xlabel(r'layer $l$')
    ax.set_ylabel(r'$\mu(X^l)$')
    plt.legend()
    plt.xscale('log')
    plt.tight_layout()
    plt.show()


def plot_transition_pred_test(sigma_w_sq_arr, max_dist, chaos_cond):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    transition_pred = sigma_w_sq_arr[np.max(np.where(chaos_cond < 1))]
    ax.plot(sigma_w_sq_arr, max_dist, label='max dist')
    ax.plot(sigma_w_sq_arr, chaos_cond, label=r'$\max (\lambda_i^p)$')
    fig.draw_without_rendering()
    ax.autoscale(False, axis="y")
    ax.plot([transition_pred, transition_pred], [-1, 2], color='black')
    ax.set_xlabel(r'$\sigma_w^2$')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_distance_matrix(dist_eq):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    x, y = np.meshgrid(np.arange(dist_eq.shape[0])+1, np.arange(dist_eq.shape[0])+1)
    # cmap = sns.diverging_palette(220, 20, s=60, as_cmap=True)
    c = ax.pcolormesh(x, y, dist_eq, cmap='Blues_r', vmin=0., vmax=np.max(dist_eq))
    ax.invert_yaxis()
    ax.set_xlabel('node')
    ax.set_ylabel('node')
    fig.colorbar(c, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_dist_block_avg(L, pred_diag, sim_diag, pred_offdiag, sim_offdiag):
    _, ax = plt.subplots(1, 1, figsize=(4, 3))
    L_arr = np.arange(L)+1
    ax.plot(L_arr, pred_diag, label='within class', color='blue')
    ax.plot(L_arr, pred_offdiag, label='across classes', color='orange')
    ax.plot(L_arr, sim_diag, linestyle='', marker='x', color='blue')
    ax.plot(L_arr, sim_offdiag, linestyle='', marker='x', color='orange')
    ax.set_xscale('log')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_gen_error_hm(gen_error, sigma_w_sq_arr, L_arr, label='gen error'):
    fig, ax = plt.subplots(1, 1, figsize=(4, 3))
    x, y = np.meshgrid(sigma_w_sq_arr, L_arr)
    c = ax.pcolormesh(x, y, gen_error.T, cmap='Blues_r')
    ax.set_xlabel(r'$\sigma_w^2$')
    ax.set_ylabel(r'$L$')
    fig.colorbar(c, ax=ax, label=label)
    plt.tight_layout()
    plt.show()


def plot_hist_crit_sigma_w_sq(crit_values):
    _, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.hist(crit_values, bins=15)
    ax.set_xlabel(r'$\sigma_{w,\mathrm{crit}}^2$')
    ax.set_title(r'Histogram of $\sigma^2_{w,\mathrm{crit}}$')
    plt.tight_layout()
    plt.show()


def plot_finite_size_gen_error(L_arr, gen_error_chaos, gen_error_oversmoothing, gen_error_crit):
    _, ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(L_arr, gen_error_chaos, marker='x', linestyle='', label='chaos')
    ax.plot(L_arr, gen_error_crit, marker='x', linestyle='', label='crit')
    ax.plot(L_arr, gen_error_oversmoothing, marker='x', linestyle='', label='oversmoothing')
    ax.set_xlabel(r'$L$')
    ax.set_ylabel('gen error')
    ax.set_xscale('log')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_KW_oversmoothing(sigma_w_sq_arr, av_node_sim_kw, std_node_sim_kw, max_K_kw, node_sim_kw,
                          av_node_sim_ds, std_node_sim_ds, max_K_ds, node_sim_ds):
    _, axs = plt.subplots(1, 2, figsize=(18*0.33, 2.1), layout='constrained')  # , figsize=(9*0.33, 2.1)
    ax1, ax2 = axs
    ax1.plot(sigma_w_sq_arr, av_node_sim_kw, color='blue')
    ax1.fill_between(sigma_w_sq_arr, av_node_sim_kw+std_node_sim_kw, av_node_sim_kw-std_node_sim_kw,
                     color='blue', alpha=0.1)
    ax1.plot(sigma_w_sq_arr, av_node_sim_ds, color='orange')
    ax1.fill_between(sigma_w_sq_arr, av_node_sim_ds+std_node_sim_ds, av_node_sim_ds-std_node_sim_ds,
                    color='orange', alpha=0.1)

    ax1.set_xlabel(r'$\sigma_w^2$')
    ax1.set_ylabel(r'$\mu(X)$ in equilibrium')

    ax2.scatter(max_K_kw.flatten(), node_sim_kw.flatten(), alpha=0.3, color='blue')
    ax2.scatter(max_K_ds.flatten(), node_sim_ds.flatten(), alpha=0.3, color='orange')
    ax2.set_xlabel(r'$\max_{\alpha}(K^{\mathrm{eq}}_{\alpha\alpha})$')
    ax2.set_ylabel(r'$\mu(X)$ in equilibrium')
    plt.show()


def plot_transition_from_numerics(sigma_w_sq_arr, oversmoothing_measure):
    _, ax = plt.subplots(1, 1, layout='constrained', figsize=(9*0.33, 2.1))
    ax.plot(sigma_w_sq_arr, oversmoothing_measure)
    ax.set_ylabel(r'$\mu(X)$ at equilibrium')
    ax.set_xlabel(r'$\sigma_w^2$')
    transition = sigma_w_sq_arr[np.min(np.where(oversmoothing_measure > 1e-5))]
    ax.vlines(x=transition, ymin=-1, ymax=1, color='red')

    gap = np.max(oversmoothing_measure) - np.min(oversmoothing_measure)
    lower = np.min(oversmoothing_measure) - 0.1*gap
    upper = np.max(oversmoothing_measure) + 0.1*gap
    ax.set_ylim((lower, upper))
    plt.show()
