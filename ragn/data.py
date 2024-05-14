import joblib
from os.path import join, splitext
from os import listdir, makedirs

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit, prange
from scipy.sparse import csr_array
from sklearn.cluster import spectral_clustering
from sklearn.preprocessing import minmax_scale

from ragn.draw import draw_ip_clusters
from ragn.utils import to_one_hot, read_graph, pairwise, set_diff
from graph_nets.utils_np import networkx_to_data_dict
from graph_nets.utils_tf import data_dicts_to_graphs_tuple


__all__ = ["init_generator"]


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
    digraph = digraph.copy()
    graph = digraph.to_undirected()
    n_nodes = graph.number_of_nodes()
    ratio_upper_bound = random_state.choice(ratio_upper_range)
    upper_bound = int(ratio_upper_bound * n_nodes)
    n_subnets = random_state.choice(np.arange(1, upper_bound if upper_bound > 1 else 2))
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
        tqdm.write("No clusters")
    return digraph


def add_edge_weigths(digraph):
    digraph = nx.convert_node_labels_to_integers(digraph)
    pos = nx.layout.spring_layout(digraph)
    weights = get_edge_weights(np.array(digraph.edges), np.stack(list(pos.values())))
    nx.set_node_attributes(digraph, pos, "pos")  # type: ignore
    nx.set_edge_attributes(digraph, dict(zip(digraph.edges, weights)), "weight")
    return digraph


def add_shortest_path(graph, random_state):
    random_state = random_state if random_state else np.random.RandomState()
    all_paths = nx.all_pairs_dijkstra(graph, weight="distance")
    end = random_state.choice(graph.nodes())
    digraph = graph.to_directed()
    solution_edges = []
    min_distance = []
    for node, (distance, path) in all_paths:
        if node != end:
            solution_edges.extend(list(pairwise(path[end])))
        min_distance.append(
            (node, dict(min_distance_to_end=distance[end], hops_to_end=len(path[end])))
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
    names = [splitext(p) for p in listdir(source_path)]
    random_state = np.random.RandomState(seed)
    for name, ext in tqdm(names):
        digraph = read_graph(join(source_path, f"{name}{ext}"), directed=True)
        digraph = add_edge_weigths(digraph)
        digraph = add_ip(digraph, random_state)
        digraph = add_shortest_path(digraph, random_state)
        joblib.dump(digraph, join(save_path, f"{name}.pickle.xz"), compress=("xz", 3))  # type: ignore
        if debug:
            draw_ip_clusters(
                digraph, save_path, name=name, ext="png", use_original_pos=True
            )


def init_generator(
    path, n_batch, scaler, random_state, seen_graphs=0, input_fields=None
):
    if scaler:
        _scaler = minmax_scale
    else:
        _scaler = None
    generator = networkx_to_graph_tuple_generator(
        batch_files_generator(
            path,
            n_batch,
            shuffle=True,
            bidim_solution=False,
            input_fields=input_fields,
            random_state=random_state,
            seen_graphs=seen_graphs,
            scaler=_scaler,
        )
    )
    return generator


def batch_files_generator(
    source_path,
    n_batch,
    shuffle=False,
    scaler=None,
    bidim_solution=True,
    input_fields=None,
    target_fields=None,
    random_state=None,
    seen_graphs=0,
    dtype=np.float32,
):
    random_state = np.random.RandomState() if random_state is None else random_state
    names = [splitext(f) for f in listdir(source_path)]
    if shuffle:
        random_state.shuffle(names)
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
            input_fields,
            target_fields,
            dtype=dtype,
            random_state=random_state,
        )
        yield input_batch, target_batch


def read_from_files(
    source_path,
    batch_names,
    scaler=None,
    bidim_solution=True,
    input_fields=None,
    target_fields=None,
    dtype=np.float32,
    random_state=None,
):
    input_batch = []
    target_batch = []
    random_state = np.random.RandomState() if random_state is None else random_state
    for name, ext in batch_names:
        digraph = read_graph(join(source_path, f"{name}{ext}"), directed=True)
        input_graph, target_graph = graph_to_input_target(
            digraph,
            scaler=scaler,
            input_fields=input_fields,
            target_fields=target_fields,
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
    input_fields=None,
    target_fields=None,
    bidim_solution=True,
    random_state=None,
):
    _graph = graph.copy()
    random_state = np.random.RandomState() if random_state is None else random_state
    input_node_fields = (
        input_fields["node"] if input_fields and "node" in input_fields else ("pos",)
    )
    input_edge_fields = (
        input_fields["edge"]
        if input_fields and "edge" in input_fields
        else ("ip", "distance")
    )
    target_node_fields = (
        target_fields["node"]
        if target_fields and "node" in target_fields
        else ("min_distance_to_end", "hops_to_end")
    )
    target_edge_fields = (
        target_fields["edge"]
        if target_fields and "edge" in target_fields
        else ("solution",)
    )
    if scaler is not None:
        d_distance = nx.get_edge_attributes(_graph, "weight")
        d_pos = nx.get_node_attributes(_graph, "pos")
        all_distance = list(d_distance.values())
        all_pos = list(d_pos.values())
        nx.set_edge_attributes(
            _graph, dict(zip(d_distance, scaler(all_distance))), "weight"
        )
        nx.set_node_attributes(_graph, dict(zip(d_pos, scaler(all_pos))), "pos")
    input_graph = _graph.copy()
    target_graph = _graph.copy()
    destination = _graph.graph["target"]
    destination_out_degree = _graph.out_degree(destination)
    destination_interface_idx = random_state.choice(range(destination_out_degree))
    destination_interface = list(_graph.out_edges(destination, data="ip"))[
        destination_interface_idx
    ][-1].astype(dtype)
    for node_index, node_feature in _graph.nodes(data=True):
        input_node_features = create_feature(
            node_feature, input_node_fields, dtype, node_index, _graph
        )
        target_node_features = create_feature(
            node_feature, target_node_fields, dtype, node_index, _graph
        )
        input_graph.add_node(node_index, features=input_node_features)
        target_graph.add_node(node_index, features=target_node_features)
    for sender, receiver, edge_feature in _graph.edges(data=True):
        input_edge_features = create_feature(edge_feature, input_edge_fields, dtype)
        input_graph.add_edge(sender, receiver, features=input_edge_features)
        if bidim_solution:
            target_edge = to_one_hot(
                create_feature(edge_feature, target_edge_fields, dtype).astype(  # type: ignore
                    np.int32
                ),
                2,
            )[0]
        else:
            target_edge = create_feature(edge_feature, target_edge_fields, dtype)
        target_graph.add_edge(sender, receiver, features=target_edge)
    input_graph.graph["features"] = destination_interface
    target_graph.graph["features"] = destination_interface
    return input_graph, target_graph


def create_feature(attr, fields, dtype, index=None, graph=None):
    features = []
    if fields == ():
        return None
    for field in fields:
        if field in attr:
            fattr = attr[field]
        elif index is not None and graph is not None:
            fattr = graph.__getattribute__(field)(index, weight="distance")
        else:
            raise ValueError
        features.append(fattr)
    return np.hstack(features).astype(dtype)


def networkx_to_graph_tuple_generator(nx_generator):
    for (
        nx_in_graphs,
        nx_gt_graphs,
    ) in nx_generator:
        gt_in_graphs = networkxs_to_graphs_tuple(nx_in_graphs)
        gt_gt_graphs = networkxs_to_graphs_tuple(nx_gt_graphs)
        yield gt_in_graphs, gt_gt_graphs


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
