import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from metrics import compute_accuracy, aggregator_path_metrics


def draw_revertion(steady_values, transient_values, top_type):
    dists_steady = {}
    dists_transient = {}
    dists_steady[r"$\sigma_c$"] = steady_values["cost"]
    dists_transient[r"$\sigma_c$"] = transient_values["cost"]
    dists_steady[r"$\sigma_l$"] = steady_values["hops"]
    dists_transient[r"$\sigma_l$"] = transient_values["hops"]
    for (k, v_steady), (_, v_transient) in zip(dists_steady.items(), dists_transient.items()):
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
            # ax.set_xlim(0, 2.55)
        elif k == r"$\pi$":
            bins = np.histogram_bin_edges(v_steady, bins="auto")
            sns.distplot(
                v_steady,
                ax=ax,
                bins=bins,
                kde=False,
                hist_kws=dict(
                    zorder=1, weights=np.full(len(v_steady), 1 / len(v_steady)), range=(0, 1)
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
                    zorder=1, weights=np.full(len(v_steady), 1 / len(v_steady)), range=(0, 1)
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
        plt.savefig(file_name + "_" + top_type + ".pdf", transparent=True)
        fig.clear()
        plt.close()
