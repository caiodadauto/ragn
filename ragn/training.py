import os
from time import time
from datetime import datetime as dt

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale

import pytop
from tqdm import tqdm

from ragn.ragn import RAGN
from ragn.utils import (
    networkx_to_graph_tuple_generator,
    get_validation_gts,
    get_signatures,
    get_accuracy,
    binary_crossentropy,
)


def log_scalars(writer, path, params, step, seen_graphs, epoch):
    with writer.as_default():
        for name, value in params.items():
            tf.summary.scalar(name, data=value, step=tf.cast(step, tf.int64))
        writer.flush()
    with open(os.path.join(path, "stopped_step.csv"), "w") as f:
        f.write("{}, {}\n".format(epoch, seen_graphs))


def save_params(file_path, **kwargs):
    import json

    with open(file_path, "w") as f:
        json.dump(kwargs, f)


def set_environment(
    tr_size,
    init_lr,
    end_lr,
    decay_steps,
    power,
    seed,
    n_batch,
    enc_conf,
    mlp_conf,
    rnn_conf,
    decision_conf,
    create_offset,
    create_scale,
    log_path,
    restore_from,
    opt,
    scale,
    n_msg,
    n_epoch,
    delta_time_to_validate,
    class_weight,
    dropped_msg_ratio,
):
    global_step = tf.Variable(0, trainable=False)
    best_val_acc_tf = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    if scale:
        scaler = minmax_scale
    else:
        scaler = None
    model = RAGN(
        enc_conf,
        mlp_conf,
        rnn_conf,
        decision_conf,
        create_offset,
        create_scale,
    )
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

    epoch = 0
    seen_graphs = 0
    status = None
    if restore_from is None:
        log_dir = os.path.join(log_path, dt.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(log_dir)
        with open(os.path.join(log_dir, "seed.csv"), "w") as f:
            f.write("{}\n".format(seed))
    else:
        log_dir = os.path.join(log_path, restore_from)
        with open(os.path.join(log_dir, "seed.csv"), "r") as f:
            seed = int(f.readline().rstrip())
        try:
            with open(os.path.join(log_dir, "stopped_step.csv"), "r") as f:
                epoch, seen_graphs = list(
                    map(lambda s: int(s), f.readline().rstrip().split(","))
                )
        except:
            pass
        print(
            "\nRestore training session from {}, "
            "stopped in epoch {} with {} processed graphs\n".format(
                log_dir, epoch, seen_graphs
            )
        )
    tf.random.set_seed(seed)
    random_state = np.random.RandomState(seed=seed)
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
    if restore_from is not None:
        status = ckpt.restore(last_ckpt_manager.latest_checkpoint)
    save_params(
        os.path.join(log_dir, "step-" + str(global_step.numpy()) + ".json"),
        tr_size=tr_size,
        n_msg=n_msg,
        epoch=epoch,
        seen_graphs=seen_graphs,
        seed=seed,
        init_lr=init_lr,
        end_lr=end_lr,
        decay_steps=decay_steps,
        power=power,
        delta_time_to_validate=delta_time_to_validate,
        class_weight=class_weight.numpy().tolist(),
        opt=opt,
        scale=scale,
        dropped_msg_ratio=dropped_msg_ratio,
        enc_conf=enc_conf,
        mlp_conf=mlp_conf,
        rnn_conf=rnn_conf,
        decision_conf=decision_conf,
        create_offset=create_offset,
        create_scale=create_scale,
    )
    return (
        model,
        lr,
        loss_fn,
        n_batch,
        optimizer,
        global_step,
        best_val_acc_tf,
        epoch,
        seen_graphs,
        log_dir,
        last_ckpt_manager,
        best_ckpt_manager,
        random_state,
        status,
        scaler,
        tf.summary.create_file_writer(
            os.path.join(log_dir, "scalar", "epoch-" + str(epoch))
        ),
    )


def init_training_generator(
    tr_path, tr_size, n_batch, scaler, random_state, seen_graphs, input_fields=None
):
    batch_bar = tqdm(
        total=tr_size,
        initial=seen_graphs,
        desc="Processed Graphs",
        leave=False,
    )
    train_generator = networkx_to_graph_tuple_generator(
        pytop.batch_files_generator(
            tr_path,
            "gpickle",
            n_batch,
            dataset_size=tr_size,
            shuffle=True,
            bidim_solution=False,
            input_fields=input_fields,
            random_state=random_state,
            seen_graphs=seen_graphs,
            scaler=scaler,
        )
    )
    return batch_bar, train_generator


def train_ragn(
    tr_size,
    tr_path,
    val_path,
    log_path,
    n_msg,
    n_epoch,
    n_batch,
    enc_conf,
    mlp_conf,
    rnn_conf,
    decision_conf,
    create_offset,
    create_scale,
    debug=False,
    seed=12345,
    init_lr=5e-3,
    end_lr=1e-5,
    decay_steps=70000,
    power=3,
    delta_time_to_validate=20,
    class_weight=[1.0, 1.0],
    restore_from=None,
    opt="adam",
    scale=False,
    dropped_msg_ratio=0.0,
    input_fields=None,
):
    def eval(in_graphs):
        output_graphs = model(in_graphs, n_msg, is_training=False)
        return output_graphs

    def update_model_weights(in_graphs, gt_graphs):
        expected = gt_graphs.edges
        with tf.GradientTape() as tape:
            output_graphs = model(in_graphs, n_msg, is_training=True)
            loss = loss_fn(expected, output_graphs, class_weight, dropped_msg_ratio)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(
            zip(gradients, model.trainable_variables), global_step=global_step
        )
        return output_graphs[-1], loss

    class_weight = tf.constant(class_weight, dtype=tf.float32)
    (
        model,
        lr,
        loss_fn,
        n_batch,
        optimizer,
        global_step,
        best_val_acc_tf,
        epoch,
        seen_graphs,
        log_dir,
        last_ckpt_manager,
        best_ckpt_manager,
        random_state,
        status,
        scaler,
        scalar_writer,
    ) = set_environment(
        tr_size,
        init_lr,
        end_lr,
        decay_steps,
        power,
        seed,
        n_batch,
        enc_conf,
        mlp_conf,
        rnn_conf,
        decision_conf,
        create_offset,
        create_scale,
        log_path,
        restore_from,
        opt,
        scale,
        n_msg,
        n_epoch,
        delta_time_to_validate,
        class_weight,
        dropped_msg_ratio,
    )

    tr_acc = None
    val_acc = None
    asserted = False
    best_val_acc = best_val_acc_tf.numpy()
    in_val_graphs, gt_val_graphs, _ = get_validation_gts(val_path, scaler, input_fields)
    epoch_bar = tqdm(total=n_epoch + epoch, initial=epoch, desc="Processed Epochs")
    epoch_bar.set_postfix(loss=None, best_val_acc=best_val_acc)
    if not debug:
        in_signarute, gt_signature = get_signatures(in_val_graphs, gt_val_graphs)
        eval = tf.function(eval, input_signature=[in_signarute])
        update_model_weights = tf.function(
            update_model_weights, input_signature=[in_signarute, gt_signature]
        )
    start_time = time()
    last_validation = start_time

    for epoch in range(epoch, n_epoch + epoch):
        batch_bar, train_generator = init_training_generator(
            tr_path, tr_size, n_batch, scaler, random_state, seen_graphs, input_fields
        )
        for in_graphs, gt_graphs, raw_edge_features in train_generator:
            n_graphs = in_graphs.n_node.shape[0]
            seen_graphs += n_graphs
            out_tr_graphs, loss = update_model_weights(in_graphs, gt_graphs)
            if not asserted and status is not None:
                status.assert_consumed()
                asserted = True
            delta_time = time() - last_validation
            if delta_time >= delta_time_to_validate:
                out_val_graphs = eval(in_val_graphs)
                last_validation = time()
                tr_acc = get_accuracy(
                    out_tr_graphs.edges.numpy(), gt_graphs.edges.numpy()
                )
                val_acc = get_accuracy(
                    out_val_graphs[-1].edges.numpy(), gt_val_graphs.edges.numpy()
                )
                val_loss = loss_fn(
                    gt_val_graphs.edges, out_val_graphs, class_weight, dropped_msg_ratio
                )
                log_scalars(
                    scalar_writer,
                    log_dir,
                    {
                        "training loss": loss.numpy(),
                        "validation loss": val_loss.numpy(),
                        "learning rate": lr().numpy(),
                        "train accuracy": tr_acc,
                        "val accuracy": val_acc,
                    },
                    global_step.numpy(),
                    seen_graphs,
                    epoch,
                )
                last_ckpt_manager.save()
                if best_val_acc <= val_acc:
                    best_ckpt_manager.save()
                    best_val_acc_tf.assign(val_acc)
                    best_val_acc = val_acc
                batch_bar.set_postfix(
                    tr_loss=loss.numpy(),
                    tr_acc=tr_acc,
                    val_loss=val_loss.numpy(),
                    val_acc=val_acc,
                )
            batch_bar.update(n_graphs)
        seen_graphs = 0
        batch_bar.close()
        epoch_bar.update()
        epoch_bar.set_postfix(tr_loss=loss.numpy(), best_val_acc=best_val_acc)
    epoch_bar.close()
