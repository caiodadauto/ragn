import itertools
from os import listdir
from os.path import splitext

import joblib
import tree
import numpy as np
import networkx as nx
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score

from graph_nets import utils_tf, utils_np
from graph_nets.graphs import NODES, EDGES, GLOBALS


__all__ = ["get_bacc", "get_f1", "get_precision", "get_bacc"]


def _compute_stacked_offsets(sizes, repeats):
    sizes = tf.cast(tf.convert_to_tensor(sizes[:-1]), tf.int32)
    offset_values = tf.cumsum(tf.concat([[0], sizes], 0))
    return utils_tf.repeat(offset_values, repeats)


def _nested_concatenate(input_graphs, field_name, axis):
    features_list = [
        getattr(gr, field_name)
        for gr in input_graphs
        if getattr(gr, field_name) is not None
    ]
    if not features_list:
        return None

    if len(features_list) < len(input_graphs):
        raise ValueError(
            "All graphs or no graphs must contain {} features.".format(field_name)
        )

    name = "concat_" + field_name
    return tree.map_structure(lambda *x: tf.concat(x, axis, name), *features_list)


def concat(
    input_graphs,
    axis,
    use_edges=True,
    use_nodes=True,
    use_globals=True,
    name="graph_concat",
):
    if not input_graphs:
        raise ValueError("List argument `input_graphs` is empty")
    utils_np._check_valid_sets_of_keys([gr._asdict() for gr in input_graphs])
    if len(input_graphs) == 1:
        return input_graphs[0]

    with tf.name_scope(name):
        if use_edges:
            edges = _nested_concatenate(input_graphs, EDGES, axis)
        else:
            edges = getattr(input_graphs[0], EDGES)
        if use_nodes:
            nodes = _nested_concatenate(input_graphs, NODES, axis)
        else:
            nodes = getattr(input_graphs[0], NODES)
        if use_globals:
            globals_ = _nested_concatenate(input_graphs, GLOBALS, axis)
        else:
            globals_ = getattr(input_graphs[0], GLOBALS)

        output = input_graphs[0].replace(nodes=nodes, edges=edges, globals=globals_)
        if axis != 0:
            return output
        n_node_per_tuple = tf.stack([tf.reduce_sum(gr.n_node) for gr in input_graphs])
        n_edge_per_tuple = tf.stack([tf.reduce_sum(gr.n_edge) for gr in input_graphs])
        offsets = _compute_stacked_offsets(n_node_per_tuple, n_edge_per_tuple)
        n_node = tf.concat(
            [gr.n_node for gr in input_graphs], axis=0, name="concat_n_node"
        )
        n_edge = tf.concat(
            [gr.n_edge for gr in input_graphs], axis=0, name="concat_n_edge"
        )
        receivers = [gr.receivers for gr in input_graphs if gr.receivers is not None]
        receivers = receivers or None
        if receivers:
            receivers = tf.concat(receivers, axis, name="concat_receivers") + offsets  # type: ignore
        senders = [gr.senders for gr in input_graphs if gr.senders is not None]
        senders = senders or None
        if senders:
            senders = tf.concat(senders, axis, name="concat_senders") + offsets  # type: ignore
        return output.replace(
            receivers=receivers, senders=senders, n_node=n_node, n_edge=n_edge
        )


def networkxs_to_graphs_tuple(
    graph_nxs, node_shape_hint=None, edge_shape_hint=None, data_type_hint=np.float32
):
    data_dicts = []
    try:
        for _graph_nx in graph_nxs:
            graph_nx = nx.DiGraph()
            graph_nx.add_nodes_from(sorted(_graph_nx.nodes(data=True)))
            graph_nx.add_edges_from(_graph_nx.edges(data=True))
            graph_nx.graph = _graph_nx.graph
            data_dict = utils_np.networkx_to_data_dict(
                graph_nx, node_shape_hint, edge_shape_hint, data_type_hint
            )
            data_dicts.append(data_dict)
    except TypeError:
        raise ValueError(
            "Could not convert some elements of `graph_nxs`. "
            "Did you pass an iterable of networkx instances?"
        )
    return utils_tf.data_dicts_to_graphs_tuple(data_dicts)


def networkx_to_graph_tuple_generator(nx_generator):
    for nx_in_graphs, nx_gt_graphs, raw_edge_features, _ in nx_generator:
        gt_in_graphs = networkxs_to_graphs_tuple(nx_in_graphs)
        gt_gt_graphs = networkxs_to_graphs_tuple(nx_gt_graphs)
        yield gt_in_graphs, gt_gt_graphs, raw_edge_features


def to_one_hot(indices, max_value):
    one_hot = np.eye(max_value)[indices]
    return one_hot


def read_graph(graph_path, directed=False):
    _, ext = splitext(graph_path)
    if ext == ".gexf":
        graph = nx.read_gexf(graph_path)
    elif ext.startswith(".pickle"):
        graph = joblib.load(graph_path)
    else:
        raise ValueError
    if directed and not nx.is_directed(graph):
        graph = nx.to_directed(graph)
    return graph


def get_signatures(in_graphs, gt_graphs):
    in_signature = utils_tf.specs_from_graphs_tuple(in_graphs, True)
    gt_signature = utils_tf.specs_from_graphs_tuple(gt_graphs, True)
    return in_signature, gt_signature


def unsorted_segment_softmax(x, idx, n_idx):
    op1 = tf.exp(x)
    op2 = tf.math.unsorted_segment_sum(op1, idx, n_idx)
    op3 = tf.gather(op2, idx)
    op4 = tf.divide(op1, op3)
    return op4


def binary_crossentropy(expected, output_graphs, entity, min_num_msg, class_weights):
    loss_for_all_msg = []
    idx_expected = tf.cast(expected, tf.int32)
    sample_weights = tf.squeeze(tf.gather(class_weights, idx_expected))
    for predicted_graphs in output_graphs[min_num_msg:]:
        predicted = predicted_graphs.__getattribute__(entity)
        msg_losses = tf.keras.losses.binary_crossentropy(expected, predicted)
        msg_losses = sample_weights * msg_losses
        msg_loss = tf.math.reduce_mean(msg_losses)
        loss_for_all_msg.append(msg_loss)
    loss = tf.math.reduce_sum(tf.stack(loss_for_all_msg))
    loss = loss / len(output_graphs)
    return loss


def parse_edges_bidim_probs(graph):
    def softmax_prob(x):
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    edges = graph.edges
    return graph.replace(edges=softmax_prob(edges))


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def set_diff(seq0, seq1):
    return list(set(seq0) - set(seq1))


def get_bacc(expected, predicted, th=0.5):
    e = expected.numpy()
    p = (predicted.numpy() >= th).astype(np.int32)
    return tf.constant(balanced_accuracy_score(e, p), dtype=tf.float32)


def get_precision(expected, predicted, th=0.5):
    e = expected.numpy()
    p = (predicted.numpy() >= th).astype(np.int32)
    return tf.constant(precision_score(e, p), dtype=tf.float32)


def get_f1(expected, predicted, th=0.5):
    e = expected.numpy()
    p = (predicted.numpy() >= th).astype(np.int32)
    return tf.constant(f1_score(e, p), dtype=tf.float32)
