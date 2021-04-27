import numpy as np
import tensorflow as tf

from graph_nets import utils_np
from graph_nets import utils_tf


def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]


def create_placeholders(batch_generator):
    input_graphs, target_graphs, _ = next(batch_generator)
    input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
    dtype = tf.as_dtype(utils_np.networkxs_to_graphs_tuple(target_graphs).edges.dtype)
    weight_ph = tf.placeholder(dtype, name="loss_weights")
    is_training_ph = tf.placeholder(tf.bool, name="training_flag")
    return input_ph, target_ph, weight_ph, is_training_ph


def create_feed_dict(
    batch_generator,
    is_training,
    weights,
    input_ph,
    target_ph,
    is_training_ph,
    weight_ph,
):
    inputs, targets, pos = next(batch_generator)
    input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
    target_graphs = utils_np.networkxs_to_graphs_tuple(targets)

    if weights[0] != 1 or weights[1] != 1:
        batch_weights = np.ones(target_graphs.edges.shape[0])
        target_args = np.argmax(target_graphs.edges, axis=-1)
        batch_weights[target_args == 0] *= weights[0]
        batch_weights[target_args == 1] *= weights[1]
    else:
        batch_weights = 1

    feed_dict = {
        input_ph: input_graphs,
        target_ph: target_graphs,
        is_training_ph: is_training,
        weight_ph: batch_weights,
    }
    return feed_dict, pos


def create_loss_ops(target_op, output_ops, weight):
    loss_ops = [
        tf.losses.softmax_cross_entropy(
            target_op.edges, output_op.edges, weights=weight
        )
        for output_op in output_ops
    ]
    return loss_ops
