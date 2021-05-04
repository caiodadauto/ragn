import os
from time import time
from datetime import datetime as dt

import numpy as np
import networkx as nx
import tensorflow as tf
import tensorflow.summary as summary
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import minmax_scale
from tensorboard.plugins.hparams import api as hp

import pytop
from tqdm import tqdm
from graph_nets import utils_np, utils_tf

from ragn.ragn import RAGN


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
            edge_scaler=scaler,
        )
    )
    return next(gt_generator)


def get_signatures(in_graphs, gt_graphs):
    in_signature = utils_tf.specs_from_graphs_tuple(in_graphs, True)
    gt_signature = utils_tf.specs_from_graphs_tuple(gt_graphs, True)
    return in_signature, gt_signature


def log_scalars(log_dir, params, step):
    with tf.summary.create_file_writer(os.path.join(log_dir, "scalars")).as_default():
        for name, value in params.items():
            tf.summary.scalar(name, data=value, step=tf.cast(step, tf.int64))


def get_accuracy(predicted, expected, th=0.5):
    float_p = predicted.numpy()
    e = expected.numpy()
    p = (float_p >= th).astype(np.int32)
    return balanced_accuracy_score(e, p)


def bi_get_accuracy(predicted, expected):
    e = expected.numpy()
    e = (e[:, 0] <= e[:, 1]).astype(int)
    p = predicted.numpy()
    p = (p[:, 0] <= p[:, 1]).astype(int)
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


def save_hp(base_dir, **kwargs):
    with summary.create_file_writer(base_dir).as_default():
        hp.hparams(kwargs)


def set_environment(
    tr_size,
    init_lr,
    end_lr,
    decay_steps,
    power,
    seed,
    n_batch,
    n_layers,
    hidden_size,
    rnn_depth,
    n_heads,
    log_path,
    restore_from,
    bidim_solution,
    opt,
    scale_edge,
    sufix_name,
    n_msg,
    n_epoch,
    delta_time_to_validate,
    class_weight,
    msg_ratio,
):
    global_step = tf.Variable(0, trainable=False)
    best_val_acc_tf = tf.Variable(0.0, trainable=False, dtype=tf.float32)

    if scale_edge:
        scaler = minmax_scale
    else:
        scaler = None

    model = RAGN(
        hidden_size=hidden_size,
        n_layers=n_layers,
        rnn_depth=rnn_depth,
        n_heads=n_heads,
        bidim=bidim_solution,
    )
    if bidim_solution:
        loss_fn = crossentropy_logists
    else:
        loss_fn = binary_crossentropy
    lr = tf.compat.v1.train.polynomial_decay(
        init_lr,
        global_step,
        decay_steps=decay_steps,
        end_learning_rate=end_lr,
        power=power,
    )
    if opt == "adam":
        optimizer = tf.compat.v1.train.AdamOptimizer(lr)
    else:
        optimizer = tf.compat.v1.train.RMSPropOptimizer(lr)

    if restore_from is None:
        if sufix_name:
            log_dir = os.path.join(
                log_path, dt.now().strftime("%Y%m%d-%H%M%S") + "-" + sufix_name
            )
        else:
            log_dir = os.path.join(log_path, dt.now().strftime("%Y%m%d-%H%M%S"))
        save_hp(
            log_dir,
            tr_size=tr_size,
            n_msg=n_msg,
            n_epoch=n_epoch,
            n_batch=n_batch,
            seed=seed,
            init_lr=init_lr,
            end_lr=end_lr,
            decay_steps=decay_steps,
            power=power,
            delta_time_to_validate=delta_time_to_validate,
            class_weight="{:.2f},{:.2f}".format(class_weight[0], class_weight[1]),
            bidim_solution=bidim_solution,
            opt=opt,
            scale_edge=scale_edge,
            msg_ratio=msg_ratio,
            n_layers=n_layers,
            hidden_size=hidden_size,
            rnn_depth=rnn_depth,
            n_heads=n_heads,
        )
    else:
        log_dir = os.path.join(log_path, restore_from)
        print("Restore training session from {}".format(log_dir))
    ckpt = tf.train.Checkpoint(
        global_step=global_step,
        optimizer=optimizer,
        model=model,
        best_val_acc_tf=best_val_acc_tf,
    )
    last_ckpt_manager = tf.train.CheckpointManager(
        ckpt, os.path.join(log_dir, "last_ckpts"), max_to_keep=3
    )
    best_ckpt_manager = tf.train.CheckpointManager(
        ckpt, os.path.join(log_dir, "best_ckpts"), max_to_keep=3
    )

    tf.random.set_seed(seed)
    random_state = np.random.RandomState(seed=seed)
    if restore_from is None:
        with open(os.path.join(log_dir, "sb.csv"), "w") as f:
            f.write("{}, {}\n".format(seed, n_batch))
        start_epoch = 0
        init_batch_tr = 1
        status = None
    else:
        with open(os.path.join(log_dir, "sb.csv"), "r") as f:
            seed, n_batch = list(
                map(lambda s: int(s), f.readline().rstrip().split(","))
            )
        tf.random.set_seed(seed)
        random_state = np.random.RandomState(seed=seed)
        status = ckpt.restore(last_ckpt_manager.latest_checkpoint)
        if global_step.numpy() > 0:
            steps_per_epoch = tr_size // n_batch
            start_epoch = global_step.numpy() // steps_per_epoch
            if start_epoch > 0:
                init_batch_tr = global_step.numpy() - steps_per_epoch * start_epoch
            else:
                init_batch_tr = global_step.numpy()
    return (
        log_dir,
        model,
        lr,
        loss_fn,
        n_batch,
        optimizer,
        global_step,
        best_val_acc_tf,
        start_epoch,
        init_batch_tr,
        last_ckpt_manager,
        best_ckpt_manager,
        random_state,
        status,
        scaler,
    )


def train_ragn(
    tr_size,
    tr_path,
    val_path,
    log_path,
    n_msg,
    n_epoch,
    n_batch,
    n_layers,
    hidden_size,
    rnn_depth,
    n_heads,
    sufix_name="",
    debug=False,
    seed=12345,
    init_lr=5e-3,
    end_lr=1e-5,
    decay_steps=70000,
    power=3,
    delta_time_to_validate=20,
    class_weight=tf.constant([1.0, 1.0], tf.float32),
    restore_from=None,
    bidim_solution=False,
    opt="adam",
    scale_edge=False,
    msg_ratio=1.0,
):
    def eval(in_graphs):
        output_graphs = model(in_graphs, n_msg, is_training=False)
        return output_graphs[-1]

    def update_model_weights(in_graphs, gt_graphs):
        expected = gt_graphs.edges
        with tf.GradientTape() as tape:
            output_graphs = model(in_graphs, n_msg, is_training=True)
            loss = loss_fn(expected, output_graphs, class_weight, msg_ratio)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables), global_step=global_step
        )
        return output_graphs[-1], loss

    (
        log_dir,
        model,
        lr,
        loss_fn,
        n_batch,
        optimizer,
        global_step,
        best_val_acc_tf,
        start_epoch,
        init_batch_tr,
        last_ckpt_manager,
        best_ckpt_manager,
        random_state,
        status,
        scaler,
    ) = set_environment(
        tr_size,
        init_lr,
        end_lr,
        decay_steps,
        power,
        seed,
        n_batch,
        n_layers,
        hidden_size,
        rnn_depth,
        n_heads,
        log_path,
        restore_from,
        bidim_solution,
        opt,
        scale_edge,
        sufix_name,
        n_msg,
        n_epoch,
        delta_time_to_validate,
        class_weight,
        msg_ratio,
    )

    tr_acc = None
    val_acc = None
    asserted = False
    best_val_acc = best_val_acc_tf.numpy()
    in_val_graphs, gt_val_graphs, _ = get_validation_gts(
        val_path, bidim_solution, scaler
    )
    in_signarute, gt_signature = get_signatures(in_val_graphs, gt_val_graphs)
    epoch_bar = tqdm(
        total=n_epoch + start_epoch, initial=start_epoch, desc="Processed Epochs"
    )
    epoch_bar.set_postfix(loss=None, best_val_acc=best_val_acc)
    if not debug:
        eval = tf.function(eval, input_signature=[in_signarute])
        update_model_weights = tf.function(
            update_model_weights, input_signature=[in_signarute, gt_signature]
        )
    start_time = time()
    last_validation = start_time
    for _ in range(start_epoch, n_epoch + start_epoch):
        batch_bar = tqdm(
            total=tr_size,
            initial=n_batch * init_batch_tr if init_batch_tr > 1 else 0,
            desc="Processed Graphs",
            leave=False,
        )
        train_generator = networkx_to_graph_tuple_generator(
            pytop.batch_files_generator(
                tr_path,
                "gpickle",
                n_batch,
                dataset_size=tr_size,
                bidim_solution=bidim_solution,
                shuffle=True,
                random_state=random_state,
                start_point=init_batch_tr,
                edge_scaler=scaler,
            )
        )
        for in_graphs, gt_graphs, raw_edge_features in train_generator:
            out_tr_graphs, loss = update_model_weights(in_graphs, gt_graphs)
            if not asserted and status is not None:
                status.assert_consumed()
                asserted = True
            delta_time = time() - last_validation
            if delta_time >= delta_time_to_validate:
                out_val_graphs = eval(in_val_graphs)
                last_validation = time()
                if bidim_solution:
                    tr_acc = bi_get_accuracy(out_tr_graphs.edges, gt_graphs.edges)
                    val_acc = bi_get_accuracy(out_val_graphs.edges, gt_val_graphs.edges)
                else:
                    tr_acc = get_accuracy(out_tr_graphs.edges, gt_graphs.edges)
                    val_acc = get_accuracy(out_val_graphs.edges, gt_val_graphs.edges)
                log_scalars(
                    log_dir,
                    {
                        "loss": loss.numpy(),
                        "learning rate": lr().numpy(),
                        "train accuracy": tr_acc,
                        "val accuracy": val_acc,
                    },
                    global_step.numpy(),
                )
                last_ckpt_manager.save()
                if best_val_acc <= val_acc:
                    best_ckpt_manager.save()
                    best_val_acc_tf.assign(val_acc)
                    best_val_acc = val_acc
            batch_bar.update(in_graphs.n_node.shape[0])
            batch_bar.set_postfix(loss=loss.numpy(), tr_acc=tr_acc, val_acc=val_acc)
        epoch_bar.update()
        epoch_bar.set_postfix(loss=loss.numpy(), best_val_acc=best_val_acc)
    epoch_bar.close()
