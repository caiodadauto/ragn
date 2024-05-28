import itertools
from os.path import basename

import joblib
import tree
import numpy as np
import networkx as nx
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score

from graph_nets import utils_tf, utils_np


__all__ = ["get_bacc", "get_f1", "get_precision", "get_bacc"]


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
    ext = ".".join(basename(graph_path).split(".")[1:])
    if ext == "gexf":
        graph = nx.read_gexf(graph_path)
    elif ext.startswith("pickle"):
        graph = joblib.load(graph_path)
    else:
        raise ValueError("Graph extension should be `gexf` or `pickle`.")
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


def unsorted_segment_norm_attention_sum(
    data, segment_ids, num_segments, dim_attention, name=None
):
    attention = data[:, :dim_attention]
    norm_attention = tf.math.unsorted_segment_sum(
        attention, segment_ids, num_segments, name=name
    )
    return norm_attention


def edge_binary_focal_crossentropy(expected, pred_graphs, min_num_msg, gamma=2, alpha=0.75):
    loss_for_all_msg = []
    for pred_graph in pred_graphs[min_num_msg:]:
        predicted = pred_graph.edges
        msg_losses = tf.keras.losses.binary_focal_crossentropy(  # type: ignore
            expected, predicted, apply_class_balancing=True, gamma=gamma, alpha=alpha
        )
        msg_loss = tf.math.reduce_mean(msg_losses)
        loss_for_all_msg.append(msg_loss)
    loss = tf.math.reduce_sum(tf.stack(loss_for_all_msg))
    loss = loss / len(pred_graphs)
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
