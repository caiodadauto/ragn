import joblib
from os.path import join
from os import listdir, makedirs

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.sparse import csr_array
from tqdm import tqdm
from numba import jit, prange
from sklearn.cluster import spectral_clustering


@jit(nopython=True, parallel=True, fastmath=True)
def get_edge_weights(edges, locations):
    size = edges.shape[0]
    weights = np.zeros(edges.shape[0])
    weights = np.ascontiguousarray(weights)
    for i in prange(size):
        p, s = edges[i]
        weights[i] = np.linalg.norm(locations[p] - locations[s])
    return weights


def get_ips(subnet_sizes, random_state, prefix_range=(20, 28)):
    prefix_size = 32
    prefix_sizes = np.zeros(len(subnet_sizes))
    prefixes = np.zeros((len(subnet_sizes), 32))
    for i, subnet_size in enumerate(subnet_sizes):
        prefix = prefixes[i].copy()
        while np.any(np.all(prefix == prefixes, axis=-1)):
            while 2 ** (32 - prefix_size) - 1 < subnet_size:
                prefix_size = random_state.choice(prefix_range)
            prefix[0:prefix_size] = random_state.choice([0, 1], size=prefix_size)
        prefixes[i] = prefix
        prefix_sizes[i] = prefix_size
    c = 0
    ips = np.zeros((subnet_sizes.sum(), 32))
    for prefix, prefix_size, subnet_size in zip(prefixes, prefix_sizes, subnet_sizes):
        suffix_size = int(32 - prefix_size)
        ips[c] = prefix.copy()
        for i in range(subnet_size):
            ip = prefix.copy()
            while np.any(
                np.all(
                    ip[-suffix_size:] == ips[c : c + subnet_size, -suffix_size:],
                    axis=-1,
                )
            ):
                ip[-suffix_size:] = random_state.choice([0, 1], size=suffix_size)
            ips[c + i, :] = ip.copy()
        c += subnet_size
    return ips, prefix_sizes


def add_ip(digraph, random_state, ratio_upper_range=(0.1, 0.2)):
    graph = digraph.to_undirected()
    n_nodes = graph.number_of_nodes()
    ratio_upper_bound = random_state.choice(ratio_upper_range)
    upper_bound = int(ratio_upper_bound * n_nodes)
    n_subnets = random_state.choice(np.arange(1, upper_bound if upper_bound > 1 else 2))
    print("nsubnets", n_subnets)
    labels = spectral_clustering(
        nx.adjacency_matrix(graph, weight="weight").toarray(), n_clusters=n_subnets
    )
    if labels is not None:
        # _, subnet_n_nodes = np.unique(labels, return_counts=True)
        subnet_n_links = np.zeros(n_subnets, dtype=int)
        for u, v in graph.edges():
            label_u = labels[u]
            label_v = labels[v]
            subnet_n_links[label_u] += 1
            subnet_n_links[label_v] += 1
        ips, prefix_sizes = get_ips(subnet_n_links, random_state)

        subnet_start_idx = np.cumsum(subnet_n_links)
        subnet_start_idx[-1] = 0
        subnet_start_idx = np.roll(subnet_start_idx, 1)
        for p, s in digraph.edges():
            label_p = labels[p]
            digraph.add_edge(
                p,
                s,
                ip=ips[subnet_start_idx[label_p]],
                prefix_size=prefix_sizes[label_p],
                cluster=label_p,
            )
            subnet_start_idx[label_p] += 1
        for n in digraph.nodes():
            label = labels[n]
            digraph.add_node(n, cluster=label)
    else:
        print("no clusters")


def add_ip_to_graphs(
    source_path: str,
    save_path: str,
    seed: int = 12345,
    debug: bool = False,
    ext: str = "pickle.xz",
) -> None:
    random_state = np.random.RandomState(seed)
    names = [p.split(".")[0] for p in listdir(source_path) if p.endswith(ext)]
    for name in tqdm(names):
        if ext == "gexf":
            digraph = nx.read_gexf(join(source_path, f"{name}.gexf"))
        elif ext.startswith("pickle"):
            digraph = joblib.load(join(source_path, f"{name}.{ext}"))
        else:
            raise ValueError
        if not nx.is_directed(digraph):
            digraph = nx.to_directed(digraph)
        print("name", name)
        add_ip(digraph, random_state)
        joblib.dump(digraph, join(save_path, f"{name}.pickle.xz"), compress=("xz", 3))  # type: ignore
        if debug:
            draw_ip_clusters(
                digraph,
                save_path=join(save_path),
                ext="png",
                name=name,
                use_original_pos=True,
            )


def add_edge_weigths_to_graphs(source_path: str, save_path: str):
    names = [
        p.split(".")[0] for p in listdir(source_path) if p.split(".")[-1] == "gexf"
    ]
    for name in tqdm(names):
        digraph = nx.read_gexf(join(source_path, f"{name}.gexf"))
        if not nx.is_directed(digraph):
            digraph = nx.to_directed(digraph)
        digraph = nx.convert_node_labels_to_integers(digraph)
        # digraph = nx.relabel_nodes(digraph, {l: np.int32(l) for l in digraph.nodes})
        pos = nx.layout.spring_layout(digraph)
        weights = get_edge_weights(
            np.array(digraph.edges), np.stack(list(pos.values()))
        )
        nx.set_node_attributes(digraph, pos, "pos")  # type: ignore
        nx.set_edge_attributes(digraph, dict(zip(digraph.edges, weights)), "weight")
        joblib.dump(digraph, join(save_path, f"{name}.pickle.xz"), compress=("xz", 3))  # type: ignore


def add_features_to_graphs(
    source_path: str, save_path: str, seed: int = 12345, debug: bool = False
):
    makedirs(save_path, exist_ok=True)
    add_edge_weigths_to_graphs(source_path, save_path)
    add_ip_to_graphs(save_path, save_path, seed=seed, debug=debug)


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


if __name__ == "__main__":
    add_features_to_graphs(
        "/home/caio/Documents/backup/Documents/university/Ph.D./projects/topology/generator/dggi_data/dggi_generation/test",
        "/home/caio/Documents/backup/Documents/university/Ph.D./projects/topology/generator/dggi_data/dggi_generation/test_features/",
        debug=True,
    )
