import joblib
from os.path import join, basename
from os import listdir, makedirs

import numpy as np
import networkx as nx
from tqdm import tqdm
from numba import jit, prange
from sklearn.cluster import spectral_clustering
# from sklearn.preprocessing import minmax_scale

from ragn.draw import draw_ip_clusters
from ragn.utils import to_one_hot, read_graph, pairwise, set_diff
from graph_nets.utils_np import networkx_to_data_dict
from graph_nets.utils_tf import data_dicts_to_graphs_tuple


__all__ = ["batch_generator_from_files"]


@jit(nopython=True, parallel=True, fastmath=True)
def get_edge_weights(edges, locations):
    size = edges.shape[0]
    weights = np.zeros(edges.shape[0])
    weights = np.ascontiguousarray(weights)
    for i in prange(size):
        p, s = edges[i]
        weights[i] = np.linalg.norm(locations[p] - locations[s])
    return weights


def get_ips(subnet_sizes, random_state, prefix_range=(12, 28)):
    prefix_size = 32
    prefix_sizes = np.zeros(len(subnet_sizes))
    prefixes = np.zeros((len(subnet_sizes), 32))
    prefix_array = np.arange(prefix_range[0], prefix_range[1] + 1)
    for i, subnet_size in enumerate(subnet_sizes):
        prefix = prefixes[i].copy()
        while np.any(np.all(prefix == prefixes, axis=-1)):
            while 2 ** (32 - prefix_size) - 1 < subnet_size:
                prefix_size = random_state.choice(prefix_array)
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
    digraph = digraph.copy()
    graph = digraph.to_undirected()
    n_nodes = graph.number_of_nodes()
    ratio_upper_bound = random_state.choice(ratio_upper_range)
    upper_bound = int(ratio_upper_bound * n_nodes)
    n_subnets = random_state.choice(np.arange(1, upper_bound if upper_bound > 1 else 2))
    labels = spectral_clustering(
        nx.adjacency_matrix(graph, weight="distance").toarray(), n_clusters=n_subnets
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
        for n in digraph.nodes():
            label = labels[n]
            prefix_size = int(prefix_sizes[label])
            digraph.add_node(
                n,
                cluster=label,
                prefix=np.pad(
                    ips[subnet_start_idx[label]][:prefix_size],
                    (0, 32 - prefix_size),
                    "constant",
                    constant_values=-1,
                ),
            )
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
    else:
        tqdm.write("No clusters")
    return digraph


def add_edge_weigths(digraph):
    digraph = nx.convert_node_labels_to_integers(digraph)
    pos = nx.layout.spring_layout(digraph)
    weights = get_edge_weights(np.array(digraph.edges), np.stack(list(pos.values())))
    nx.set_node_attributes(digraph, pos, "pos")  # type: ignore
    nx.set_edge_attributes(digraph, dict(zip(digraph.edges, weights)), "distance")
    return digraph


def add_shortest_path(digraph, random_state):
    min_distance = []
    solution_edges = []
    random_state = random_state if random_state else np.random.RandomState()
    all_paths = nx.all_pairs_dijkstra(digraph, weight="distance")
    end = random_state.choice(digraph.nodes())
    for node, (distance, paths) in all_paths:
        if node != end:
            solution_edges.extend(list(pairwise(paths[end])))
        min_distance.append(
            (node, dict(min_distance_to_end=distance[end], hops_to_end=len(paths[end])))
        )
    digraph.add_nodes_from(min_distance)
    digraph.add_edges_from(set_diff(digraph.edges(), solution_edges), solution=False)
    digraph.add_edges_from(solution_edges, solution=True)
    digraph.graph["target"] = end
    return digraph


def add_features_to_graphs(
    source_path: str, save_path: str, seed: int = 12345, debug: bool = False
):
    makedirs(save_path, exist_ok=True)
    names = [
        (basename(p).split(".")[0], ".".join(basename(p).split(".")[1:]))
        for p in listdir(source_path)
    ]
    random_state = np.random.RandomState(seed)
    for name, ext in tqdm(names):
        digraph = read_graph(join(source_path, f"{name}.{ext}"), directed=True)
        digraph = add_edge_weigths(digraph)
        digraph = add_ip(digraph, random_state)
        digraph = add_shortest_path(digraph, random_state)
        joblib.dump(digraph, join(save_path, f"{name}.pickle.xz"), compress=("xz", 3))  # type: ignore
        if debug:
            draw_ip_clusters(
                digraph, save_path, name=name, ext="png", use_original_pos=True
            )


# def init_generator(path, n_batch, scale_features, random_state, seen_graphs=0):
#     if scale_features:
#         _scaler = minmax_scale
#     else:
#         _scaler = None
#     generator = networkx_to_graph_tuple_generator(
#         batch_files_generator(
#             path,
#             n_batch,
#             shuffle=True,
#             bidim_solution=False,
#             random_state=random_state,
#             seen_graphs=seen_graphs,
#             scaler=_scaler,
#         )
#     )
#     return generator


def batch_generator_from_files(
    source_path,
    n_batch,
    shuffle=False,
    scaler=None,
    bidim_solution=True,
    random_state=None,
    seen_graphs=0,
    dtype=np.float32,
    sample=-1.0,
):
    random_state = np.random.RandomState() if random_state is None else random_state
    names = [
        (basename(p).split(".")[0], ".".join(basename(p).split(".")[1:]))
        for p in listdir(source_path)
    ]
    if shuffle:
        random_state.shuffle(names)
    if sample > 0 and sample < 1:
        names = random_state.choice(names, int(len(names) * sample), replace=False)
    if seen_graphs > 0:
        names = names[seen_graphs + 1 :]
    num_graphs = len(names)
    if n_batch > 0:
        slices = np.arange(0, num_graphs, n_batch)
        slices[-1] = num_graphs
    else:
        slices = np.array([0, num_graphs])
    for i in range(1, len(slices)):
        batch_names = names[slices[i - 1] : slices[i]]
        input_batch, target_batch = read_from_files(
            source_path,
            batch_names,
            scaler,
            bidim_solution,
            dtype=dtype,
            random_state=random_state,
        )
        gt_in_graphs = networkxs_to_graphs_tuple(input_batch)
        gt_gt_graphs = networkxs_to_graphs_tuple(target_batch)
        yield gt_in_graphs, gt_gt_graphs


def read_from_files(
    source_path,
    batch_names,
    scaler=None,
    bidim_solution=True,
    dtype=np.float32,
    random_state=None,
):
    input_batch = []
    target_batch = []
    random_state = np.random.RandomState() if random_state is None else random_state
    for name, ext in batch_names:
        digraph = read_graph(join(source_path, f"{name}.{ext}"), directed=True)
        input_graph, target_graph = graph_to_input_target(
            digraph,
            scaler=scaler,
            bidim_solution=bidim_solution,
            dtype=dtype,
            random_state=random_state,
        )
        input_batch.append(input_graph)
        target_batch.append(target_graph)
    return input_batch, target_batch


def graph_to_input_target(
    graph,
    dtype=np.float32,
    scaler=None,
    bidim_solution=False,
    random_state=None,
):
    _graph = graph.copy()
    random_state = np.random.RandomState() if random_state is None else random_state
    if scaler is not None:
        d_distance = nx.get_edge_attributes(_graph, "distance")
        d_pos = nx.get_node_attributes(_graph, "pos")
        all_distance = list(d_distance.values())
        all_pos = list(d_pos.values())
        nx.set_edge_attributes(
            _graph, dict(zip(d_distance, scaler(all_distance))), "distance"
        )
        nx.set_node_attributes(_graph, dict(zip(d_pos, scaler(all_pos))), "pos")
    input_graph = _graph.copy()
    target_graph = _graph.copy()
    for node_index, node_feature in _graph.nodes(data=True):
        input_node_features = create_feature(node_feature, ("prefix",), dtype)
        target_node_features = create_feature(
            node_feature, ("min_distance_to_end", "hops_to_end"), dtype
        )
        input_graph.add_node(node_index, features=input_node_features)
        target_graph.add_node(node_index, features=target_node_features)
    for sender, receiver, edge_feature in _graph.edges(data=True):
        input_edge_features = create_feature(edge_feature, ("ip", "distance"), dtype)
        input_graph.add_edge(sender, receiver, features=input_edge_features)
        if bidim_solution:
            target_edge = to_one_hot(
                create_feature(edge_feature, ("solution",), dtype).astype(np.int32),
                2,
            )[0]
        else:
            target_edge = create_feature(edge_feature, ("solution",), dtype)
        target_graph.add_edge(sender, receiver, features=target_edge)
    input_graph.graph["features"] = np.int32(_graph.graph["target"])
    target_graph.graph["features"] = np.int32(_graph.graph["target"])
    return input_graph, target_graph


def create_feature(attr, fields, dtype):
    features = []
    for field in fields:
        if field in attr:
            fattr = attr[field]
        else:
            raise ValueError
        features.append(fattr)
    return np.hstack(features).astype(dtype)


# def networkx_to_graph_tuple_generator(nx_generator):
#     for nx_in_graphs, nx_gt_graphs in nx_generator:
#         gt_in_graphs = networkxs_to_graphs_tuple(nx_in_graphs)
#         gt_gt_graphs = networkxs_to_graphs_tuple(nx_gt_graphs)
#         yield gt_in_graphs, gt_gt_graphs


def networkxs_to_graphs_tuple(
    graph_nxs, node_shape_hint=None, edge_shape_hint=None, data_type_hint=np.float32
):
    data_dicts = []
    try:
        for graph_nx in graph_nxs:
            data_dict = networkx_to_data_dict(
                graph_nx, node_shape_hint, edge_shape_hint, data_type_hint
            )
            data_dicts.append(data_dict)
    except TypeError:
        raise ValueError(
            "Could not convert some elements of `graph_nxs`. "
            "Did you pass an iterable of networkx instances?"
        )
    return data_dicts_to_graphs_tuple(data_dicts)
