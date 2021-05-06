import os
from time import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import minmax_scale

import pytop
from tqdm import tqdm

from ragn.ragn import RAGN
from ragn.policy import get_stages
from ragn.draw import draw_acc, draw_revertion
from ragn.utils import (
    networkx_to_graph_tuple_generator,
    get_validation_gts,
    get_signatures,
    compute_dist_bacc,
    to_numpy,
)


def set_environment(
    restore_from,
    n_layers,
    hidden_size,
    rnn_depth,
    n_heads,
    log_path,
    bidim_solution,
    scale_edge,
):
    if scale_edge:
        scaler = minmax_scale
    else:
        scaler = None
    log_dir = os.path.join(log_path, restore_from)
    best_dir = os.path.join(log_dir, "best_ckpts")

    with open(os.path.join(log_dir, "sb.csv"), "r") as f:
        seed, n_batch = list(map(lambda s: int(s), f.readline().rstrip().split(",")))
    tf.random.set_seed(seed)
    random_state = np.random.RandomState(seed=seed)

    global_step = tf.Variable(0, trainable=False)
    best_val_acc_tf = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    model = RAGN(
        hidden_size=hidden_size,
        n_layers=n_layers,
        rnn_depth=rnn_depth,
        n_heads=n_heads,
        bidim=bidim_solution,
    )
    ckpt = tf.train.Checkpoint(
        global_step=global_step, model=model, best_val_acc_tf=best_val_acc_tf
    )
    manager = tf.train.CheckpointManager(ckpt, best_dir, max_to_keep=3)
    _ = ckpt.restore(manager.latest_checkpoint).expect_partial()
    print(
        "\nModel state restored from {} trained until step {} with acc {}\n".format(
            best_dir, global_step.num(), best_val_acc_tf.numpy()
        )
    )
    return (
        model,
        n_batch,
        global_step,
        best_val_acc_tf,
        random_state,
        scaler,
    )


def train_ragn(
    test_path,
    log_path,
    restore_from,
    n_msg,
    n_layers,
    hidden_size,
    rnn_depth,
    n_heads,
    debug=False,
    bidim_solution=False,
    scale_edge=False,
    # class_weight=tf.constant([1.0, 1.0], tf.float32),
):
    def eval(in_graphs):
        output_graphs = model(in_graphs, n_msg, is_training=False)
        return output_graphs[-1]

    (
        model,
        n_batch,
        global_step,
        best_val_acc_tf,
        random_state,
        scaler,
    ) = set_environment(
        restore_from,
        n_layers,
        hidden_size,
        rnn_depth,
        n_heads,
        log_path,
        bidim_solution,
        scale_edge,
    )
    in_test_graphs, gt_test_graphs, _ = get_validation_gts(
        test_path, bidim_solution, scaler
    )
    in_signarute, gt_signature = get_signatures(in_test_graphs, gt_test_graphs)
    if not debug:
        eval = tf.function(eval, input_signature=[in_signarute])
    out_test_graphs = eval(in_test_graphs)

    in_test_graphs = to_numpy(in_test_graphs), 
    gt_test_graphs = to_numpy(gt_test_graphs), 
    out_test_graphs = to_numpy(out_test_graphs), 
    test_dist_acc = compute_dist_bacc(out_test_graphs, gt_test_graphs, bidim_solution)
    stages = get_stages(in_test_graphs, gt_test_graphs, out_test_graphs)
    draw_revertion(stages["steady"], stages["transient"])
    draw_acc(test_dist_acc)

