import tree
import numpy as np
import networkx as nx
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score

import pytop
from tqdm import tqdm
from graph_nets import utils_tf, utils_np
from graph_nets.graphs import NODES, EDGES, GLOBALS


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
            "All graphs or no graphs must contain {} features.".format(
                field_name)
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

        output = input_graphs[0].replace(
            nodes=nodes, edges=edges, globals=globals_)
        if axis != 0:
            return output
        n_node_per_tuple = tf.stack(
            [tf.reduce_sum(gr.n_node) for gr in input_graphs])
        n_edge_per_tuple = tf.stack(
            [tf.reduce_sum(gr.n_edge) for gr in input_graphs])
        offsets = _compute_stacked_offsets(n_node_per_tuple, n_edge_per_tuple)
        n_node = tf.concat(
            [gr.n_node for gr in input_graphs], axis=0, name="concat_n_node"
        )
        n_edge = tf.concat(
            [gr.n_edge for gr in input_graphs], axis=0, name="concat_n_edge"
        )
        receivers = [
            gr.receivers for gr in input_graphs if gr.receivers is not None]
        receivers = receivers or None
        if receivers:
            receivers = tf.concat(
                receivers, axis, name="concat_receivers") + offsets
        senders = [gr.senders for gr in input_graphs if gr.senders is not None]
        senders = senders or None
        if senders:
            senders = tf.concat(senders, axis, name="concat_senders") + offsets
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


def init_generator(
    path, n_batch, scaler, random_state, seen_graphs=0, size=None, input_fields=None
):
    if size is not None:
        batch_bar = tqdm(
            total=size,
            initial=seen_graphs,
            desc="Processed Graphs",
            leave=False,
        )
    generator = networkx_to_graph_tuple_generator(
        pytop.batch_files_generator(
            path,
            "gpickle",
            n_batch,
            dataset_size=size,
            shuffle=True,
            bidim_solution=False,
            input_fields=input_fields,
            random_state=random_state,
            seen_graphs=seen_graphs,
            scaler=scaler,
        )
    )
    if size is not None:
        return batch_bar, generator
    else:
        return None, generator


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


def get_accuracy(predicted, expected, bidim=False, th=0.5):
    if bidim:
        e = (expected[:, 0] <= expected[:, 1]).astype(int)
        p = (predicted[:, 0] <= predicted[:, 1]).astype(int)
    else:
        e = expected
        p = (predicted >= th).astype(np.int32)
    return balanced_accuracy_score(e, p)


def binary_crossentropy(output_graphs, expected,  class_weight, ratio):
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
    if start_idx == len(output_graphs):
        start_idx = len(output_graphs) - 1
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


def compute_dist_bacc(predicted, ground_truth):
    n_graphs = predicted.n_node.shape[0]
    accs = np.zeros(n_graphs)
    for idx in range(n_graphs):
        pred_graph = utils_np.get_graph(predicted, idx)
        gt_graph = utils_np.get_graph(ground_truth, idx)
        acc = get_accuracy(pred_graph.edges, gt_graph.edges, bidim=False)
        accs[idx] = acc
    return accs


def parse_edges_bidim_probs(graph):
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
