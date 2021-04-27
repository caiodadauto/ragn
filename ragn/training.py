import os
from time import time
from datetime import datetime as dt

import numpy as np
import sonnet as snt
import networkx as nx
import tensorflow as tf
from sklearn.metrics import balanced_accuracy_score

import pytop
from tqdm import tqdm
from graph_nets import utils_np, utils_tf

# from draw import draw_revertion
from ragn.ragn import EncodeProcessDecode


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


def get_validation_gts(path):
    gt_generator = networkx_to_graph_tuple_generator(
        pytop.batch_files_generator(
            path,
            "gpickle",
            -1,
            bidim_solution=False,
        )
    )
    return next(gt_generator)


def get_signatures(in_graphs, gt_graphs):
    in_signature = utils_tf.specs_from_graphs_tuple(in_graphs, True)
    gt_signature = utils_tf.specs_from_graphs_tuple(gt_graphs, True)
    return in_signature, gt_signature


def log_scalars(writer, params, step):
    with writer.as_default():
        for name, value in params.items():
            tf.summary.scalar(name, data=value, step=tf.cast(step, tf.int64))


def get_accuracy(predicted, expected, th=0.5):
    float_p = predicted.numpy()
    e = expected.numpy()
    p = (float_p >= th).astype(np.int32)
    return balanced_accuracy_score(e, p)


def train_ragn(
    tr_size,
    tr_path,
    val_path,
    log_path,
    n_msg,
    n_epoch,
    n_batch,
    debug=False,
    seed=12345,
    init_lr=5e-3,
    end_lr=1e-5,
    decay_steps=70000,
    power=3,
    delta_time_to_validate=20,
    class_weight=tf.constant([1.0, 1.0], tf.float32),
):
    def eval(in_graphs):
        output_graphs = model(in_graphs, n_msg, is_training=False)
        return output_graphs[-1]

    def update_model_weights(in_graphs, gt_graphs):
        loss_for_all_msg = []
        expected = gt_graphs.edges
        with tf.GradientTape() as tape:
            output_graphs = model(in_graphs, n_msg, is_training=True)
            for predicted_graphs in output_graphs:
                predicted = predicted_graphs.edges
                losses = tf.keras.losses.binary_crossentropy(expected, predicted)
                losses = tf.gather(class_weight, tf.cast(expected, tf.int32)) * losses
                loss_for_all_msg.append(tf.math.reduce_mean(losses))
            loss = tf.math.reduce_sum(tf.stack(loss_for_all_msg))
            loss = loss / len(output_graphs)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables), global_step=global_step
        )
        return output_graphs[-1], loss

    tf.random.set_seed(seed)
    random_state = np.random.RandomState(seed=seed)
    global_step = tf.Variable(0, trainable=False)
    logdir = os.path.join(log_path, dt.now().strftime("%Y%m%d-%H%M%S"))
    scalar_writer = tf.summary.create_file_writer(os.path.join(logdir + "/scalars"))

    model = EncodeProcessDecode()
    in_val_graphs, gt_val_graphs, _ = get_validation_gts(val_path)
    in_signarute, gt_signature = get_signatures(in_val_graphs, gt_val_graphs)
    lr = tf.compat.v1.train.polynomial_decay(
        init_lr,
        global_step,
        decay_steps=decay_steps,
        end_learning_rate=end_lr,
        power=power,
    )
    optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)
    epoch_bar = tqdm(total=n_epoch, desc="Processed Epochs")

    ckpt = tf.train.Checkpoint(step=global_step, optimizer=optimizer, net=model)
    last_ckpt_manager = tf.train.CheckpointManager(
        ckpt, os.path.join(logdir, "last_ckpts"), max_to_keep=3
    )
    best_ckpt_manager = tf.train.CheckpointManager(
        ckpt, os.path.join(logdir, "best_ckpts"), max_to_keep=3
    )

    if not debug:
        eval = tf.function(eval, input_signature=[in_signarute])
        update_model_weights = tf.function(
            update_model_weights, input_signature=[in_signarute, gt_signature]
        )

    tr_acc = None
    val_acc = None
    best_val_acc = 0
    start_time = time()
    last_validation = start_time
    for _ in range(n_epoch):
        batch_bar = tqdm(total=tr_size, desc="Processed Graphs", leave=False)
        print("Get training batch")
        train_generator = networkx_to_graph_tuple_generator(
            pytop.batch_files_generator(
                tr_path,
                "gpickle",
                n_batch,
                dataset_size=tr_size,
                bidim_solution=False,
                shuffle=True,
                random_state=random_state,
            )
        )
        for in_graphs, gt_graphs, raw_edge_features in train_generator:
            out_tr_graphs, loss = update_model_weights(in_graphs, gt_graphs)
            log_scalars(
                scalar_writer,
                {"loss": loss.numpy(), "learning rate": lr().numpy()},
                global_step.numpy(),
            )
            delta_time = time() - last_validation
            if delta_time >= delta_time_to_validate:
                out_val_graphs = eval(in_val_graphs)
                last_validation = time()
                tr_acc = get_accuracy(out_tr_graphs.edges, gt_graphs.edges)
                val_acc = get_accuracy(out_val_graphs.edges, gt_val_graphs.edges)
                log_scalars(
                    scalar_writer,
                    {"train accuracy": tr_acc, "val accuracy": val_acc},
                    global_step,
                )
                last_ckpt_manager.save()
                if best_val_acc <= val_acc:
                    best_ckpt_manager.save()
            batch_bar.update(in_graphs.n_node.shape[0])
            batch_bar.set_postfix(loss=loss.numpy(), tr_acc=tr_acc, val_acc=val_acc)
        epoch_bar.update()
        epoch_bar.set_postfix(loss=loss, best_val_acc=best_val_acc)
    epoch_bar.close()
