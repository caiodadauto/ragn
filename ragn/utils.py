import numpy as np
import networkx as nx
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score

import pytop
from graph_nets import utils_np, utils_tf


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


def get_validation_gts(path, bidim_solution, scaler):
    gt_generator = networkx_to_graph_tuple_generator(
        pytop.batch_files_generator(
            path,
            "gpickle",
            -1,
            bidim_solution=bidim_solution,
            scaler=scaler,
            input_fields=dict(node=("ip", "pos")),
        )
    )
    return next(gt_generator)


def get_signatures(in_graphs, gt_graphs):
    in_signature = utils_tf.specs_from_graphs_tuple(in_graphs, True)
    gt_signature = utils_tf.specs_from_graphs_tuple(gt_graphs, True)
    return in_signature, gt_signature


def get_accuracy(predicted, expected, th=0.5):
    p = (predicted >= th).astype(np.int32)
    return balanced_accuracy_score(expected, p)


def bi_get_accuracy(predicted, expected):
    e = (expected[:, 0] <= expected[:, 1]).astype(int)
    p = (predicted[:, 0] <= predicted[:, 1]).astype(int)
    return balanced_accuracy_score(e, p)


def binary_crossentropy(expected, output_graphs, class_weight, ratio):
    loss_for_all_msg = []
    start_idx = int(np.ceil(len(output_graphs) * ratio))
    for predicted_graphs in output_graphs[start_idx:]:
        predicted = predicted_graphs.edges
        msg_losses = tf.keras.losses.binary_crossentropy(expected, predicted)
        msg_losses = tf.gather(class_weight, tf.cast(expected, tf.int32)) * msg_losses
        msg_loss = tf.math.reduce_mean(msg_losses)
        loss_for_all_msg.append(msg_loss)
    loss = tf.math.reduce_sum(tf.stack(loss_for_all_msg))
    loss = loss / len(output_graphs)
    return loss


def crossentropy_logists(expected, output_graphs, class_weight, ratio):
    loss_for_all_msg = []
    start_idx = int(np.ceil(len(output_graphs) * ratio))
    for predicted_graphs in output_graphs[start_idx:]:
        predicted = predicted_graphs.edges
        msg_loss = tf.compat.v1.losses.softmax_cross_entropy(
            expected,
            predicted,
            tf.gather(class_weight, tf.cast(expected[:, 1] == 1, tf.int32)),
        )
        loss_for_all_msg.append(msg_loss)
    loss = tf.math.reduce_sum(tf.stack(loss_for_all_msg))
    loss = loss / len(output_graphs)
    return loss


def mse(expected, output_graphs, ratio):
    loss_for_all_msg = []
    start_idx = int(np.ceil(len(output_graphs) * ratio))
    for predicted_graphs in output_graphs[start_idx:]:
        predicted = predicted_graphs.nodes
        msg_loss = tf.metrics.mse(expected, predicted)
        loss_for_all_msg.append(msg_loss)
    loss = tf.math.reduce_sum(tf.stack(loss_for_all_msg))
    loss = loss / len(output_graphs)
    return loss


def bi_loss(true_graphs, output_graphs, class_weight, ratio):
    return crossentropy_logists(
        true_graphs.edges, output_graphs, class_weight, ratio
    ) + mse(true_graphs.nodes, output_graphs, ratio)


def compute_dist_bacc(predicted, ground_truth, bidim):
    n_graphs = predicted.n_node.shape[0]
    accs = np.zeros(n_graphs)
    for idx in range(n_graphs):
        pred_graph = utils_np.get_graph(predicted, idx)
        gt_graph = utils_np.get_graph(predicted, idx)
        if bidim:
            acc = bi_get_accuracy(pred_graph.edges, gt_graph.edges)
        else:
            acc = get_accuracy(pred_graph.edges, gt_graph.edges)
        accs[idx] = acc
    return accs


def parse_edges_bi_probs(graph):
    def softmax_prob(x):
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    edges = graph.edges
    return graph.replace(edges=softmax_prob(edges))


def to_numpy(g):
    return g.replace(
        edges=g.edges.numpy(),
        nodes=g.nodes.numpy(),
        globals=g.globals.numpy(),
        receivers=g.receivers.numpy(),
        senders=g.senders.numpy(),
        n_node=g.n_node.numpy(),
        n_edge=g.n_edge.numpy(),
    )
