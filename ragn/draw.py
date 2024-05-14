import os
from os.path import join

import numpy as np
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt


def draw_acc(accs, log_path, title):
    path = os.path.join(log_path, "img")
    os.makedirs(path, exist_ok=True)
    sns.set_style("ticks")
    fig = plt.figure(dpi=300)
    ax = fig.subplots(1, 1, sharey=False)
    sns.histplot(
        accs,
        ax=ax,
        # bins=5,
        kde=True,
        cumulative=True,
        stat="density",
        binrange=(0, 1),
        label=r"Zoo, {} = {:.3f}".format(r"$\overline{ACC}$", accs.mean()),
        kde_kws=dict(cumulative=True),
    )
    ax.set_xlabel(r"ACC")
    ax.set_ylabel("Cumulative Frequency")
    ax.legend()
    ax.set_yticks(np.arange(0, 1.25, 0.25))
    ax.yaxis.grid(True)
    ax.set_title(title)
    fig.tight_layout()
    plt.savefig(os.path.join(path, "acc.pdf"), transparent=True)
    fig.clear()
    plt.close()


def draw_revertion(steady_values, transient_values, log_path):
    path = os.path.join(log_path, "img")
    os.makedirs(path, exist_ok=True)
    dists_steady = {}
    dists_transient = {}
    dists_steady[r"$\sigma_c$"] = steady_values["cost"]
    dists_transient[r"$\sigma_c$"] = transient_values["cost"]
    dists_steady[r"$\sigma_l$"] = steady_values["hops"]
    dists_transient[r"$\sigma_l$"] = transient_values["hops"]
    for (k, v_steady), (_, v_transient) in zip(
        dists_steady.items(), dists_transient.items()
    ):
        fig = plt.figure(dpi=300)
        ax = fig.subplots(1, 1, sharey=False)
        if k == r"$\pi$":
            lmean = r"$\overline{\pi}$"
            file_name = "pi"
        elif k == r"$\sigma_c$":
            lmean = r"$\overline{\sigma_c}$"
            file_name = "sigma_c"
        else:
            lmean = r"$\overline{\sigma_l}$"
            file_name = "sigma_l"
        if k == r"$\sigma_l$":
            bins = np.histogram_bin_edges(v_steady, bins="auto")
            sns.distplot(
                v_steady,
                ax=ax,
                bins=bins,
                kde=False,
                hist_kws=dict(
                    zorder=1,
                    weights=np.full(len(v_steady), 1 / len(v_steady)),
                    range=(0, v_steady.max()),
                ),
                label="Steady State, {} = {:.3f}".format(lmean, v_steady.mean()),
            )
            bins = np.histogram_bin_edges(v_transient, bins="auto")
            sns.distplot(
                v_transient,
                ax=ax,
                bins=bins,
                kde=False,
                hist_kws=dict(
                    zorder=0,
                    hatch="//",
                    weights=np.full(len(v_transient), 1 / len(v_transient)),
                    range=(0, v_transient.max()),
                ),
                label="Transient, {} = {:.3f}".format(lmean, v_transient.mean()),
            )
        elif k == r"$\pi$":
            bins = np.histogram_bin_edges(v_steady, bins="auto")
            sns.distplot(
                v_steady,
                ax=ax,
                bins=bins,
                kde=False,
                hist_kws=dict(
                    zorder=1,
                    weights=np.full(len(v_steady), 1 / len(v_steady)),
                    range=(0, 1),
                ),
                label="Brite, {} = {:.3f}".format(lmean, v_steady.mean()),
            )
            bins = np.histogram_bin_edges(v_transient, bins="auto")
            sns.distplot(
                v_transient,
                ax=ax,
                kde=False,
                bins=bins,
                hist_kws=dict(
                    zorder=0,
                    hatch="//",
                    weights=np.full(len(v_transient), 1 / len(v_transient)),
                    range=(0, 1),
                ),
                label="Transient, {} = {:.3f}".format(lmean, v_transient.mean()),
            )
        else:
            bins = np.histogram_bin_edges(v_steady, bins="auto")
            sns.distplot(
                v_steady,
                ax=ax,
                bins=10,
                kde=False,
                hist_kws=dict(
                    zorder=1,
                    weights=np.full(len(v_steady), 1 / len(v_steady)),
                    range=(0, 1),
                ),
                label="Steady State, {} = {:.3f}".format(lmean, v_steady.mean()),
            )
            bins = np.histogram_bin_edges(v_transient, bins="auto")
            sns.distplot(
                v_transient,
                ax=ax,
                bins=10,
                kde=False,
                hist_kws=dict(
                    zorder=0,
                    hatch="//",
                    weights=np.full(len(v_transient), 1 / len(v_transient)),
                    range=(0, 1),
                ),
                label="Transient, {} = {:.3f}".format(lmean, v_transient.mean()),
            )
            ax.set_xlim(0, 1.05)
        if k == r"$\pi$":
            ax.set_ylabel("Percentage of Topologies", fontsize=14)
            yticks = list(np.arange(0, 1, 0.15))
            # yticks[-1] = 1.00
            ax.set_yticks(yticks)
        else:
            ax.set_ylabel("Percentage of Messages", fontsize=14)
            ax.set_yticks(np.arange(0, 0.9, 0.15))
        ax.yaxis.grid(True)
        ax.set_xlabel(k, fontsize=18)
        ax.legend(loc="upper left", prop={"size": 14})
        ax.tick_params(axis="x", labelsize=14)
        ax.tick_params(axis="y", labelsize=14)
        fig.tight_layout()
        plt.savefig(os.path.join(path, file_name + ".pdf"), transparent=True)
        fig.clear()
        plt.close()


def draw_ip_clusters(digraph, save_path, name="", ext="pdf", use_original_pos=False):
    if use_original_pos:
        pos = list(dict(digraph.nodes(data="pos")).values())
    else:
        pos = nx.spring_layout(digraph)
    node_colors = list(dict(digraph.nodes(data="cluster")).values())
    edge_colors = [c for _, _, c in list(digraph.edges(data="cluster"))]
    nx.draw_networkx_nodes(digraph, pos=pos, node_color=node_colors)  # type: ignore
    nx.draw_networkx_edges(
        digraph, pos=pos, connectionstyle="arc3,rad=0.2", edge_color=edge_colors  # type: ignore
    )
    if name == "":
        plt.savefig(join(save_path, f"ip_cluster.{ext}"))
    else:
        plt.savefig(join(save_path, f"ip_cluster_{name}.{ext}"))
    plt.close()
