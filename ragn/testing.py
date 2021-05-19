from ragn.utils import (
    networkx_to_graph_tuple_generator,
    get_validation_gts,
    get_signatures,
    compute_dist_bacc,
    to_numpy,
)
from ragn.draw import draw_acc, draw_revertion
from ragn.policy import get_stages
from ragn.ragn import RAGN
from tqdm import tqdm
import pytop
from sklearn.preprocessing import minmax_scale
import tensorflow as tf
import numpy as np
import os
from time import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def set_environment(
    restore_from,
    n_layers,
    hidden_size,
    rnn_depth,
    n_heads,
    n_att,
    create_offset,
    create_scale,
    log_path,
    bidim_solution,
    scale,
):
    global_step = tf.Variable(0, trainable=False)
    best_val_acc_tf = tf.Variable(0.0, trainable=False, dtype=tf.float32)
    if scale:
        scaler = minmax_scale
    else:
        scaler = None
    log_dir = os.path.join(log_path, restore_from)
    best_dir = os.path.join(log_dir, "best_ckpts")
    model = RAGN(
        hidden_size=hidden_size,
        n_layers=n_layers,
        rnn_depth=rnn_depth,
        n_heads=n_heads,
        n_att=n_att,
        create_offset=create_offset,
        create_scale=create_scale,
        bidim=bidim_solution,
    )

    with open(os.path.join(log_dir, "seed.csv"), "r") as f:
        seed = int(f.readline().rstrip())
    try:
        with open(os.path.join(log_dir, "stopped_step.csv"), "r") as f:
            epoch, seen_graphs = list(
                map(lambda s: int(s), f.readline().rstrip().split(","))
            )
    except:
        pass
    tf.random.set_seed(seed)
    random_state = np.random.RandomState(seed=seed)
    ckpt = tf.train.Checkpoint(
        global_step=global_step, model=model, best_val_acc_tf=best_val_acc_tf
    )
    manager = tf.train.CheckpointManager(ckpt, best_dir, max_to_keep=3)
    _ = ckpt.restore(manager.latest_checkpoint).expect_partial()
    print(
        "\nRestore model session from {}, "
        "stoped in epoch {} with {} processed "
        "graphs and presenting a best validation accuracy of {}\n".format(
            log_dir, epoch, seen_graphs, best_val_acc_tf.numpy()
        )
    )
    return (
        model,
        global_step,
        random_state,
        scaler,
    )


def test_ragn(
    test_path,
    log_path,
    restore_from,
    n_msg,
    n_layers,
    hidden_size,
    rnn_depth,
    n_heads,
    n_att,
    create_offset,
    create_scale,
    debug=False,
    bidim_solution=True,
    scale=False,
    input_fields=None,
):
    def eval(in_graphs):
        output_graphs = model(in_graphs, n_msg, is_training=False)
        return output_graphs[-1]

    (
        model,
        global_step,
        random_state,
        scaler,
    ) = set_environment(
        restore_from,
        n_layers,
        hidden_size,
        rnn_depth,
        n_heads,
        n_att,
        create_offset,
        create_scale,
        log_path,
        bidim_solution,
        scale,
    )
    in_test_graphs, gt_test_graphs, _ = get_validation_gts(
        test_path, bidim_solution, scaler, input_fields
    )
    in_signarute, gt_signature = get_signatures(in_test_graphs, gt_test_graphs)
    if not debug:
        eval = tf.function(eval, input_signature=[in_signarute])
    out_test_graphs = eval(in_test_graphs)

    in_test_graphs = to_numpy(in_test_graphs)
    gt_test_graphs = to_numpy(gt_test_graphs)
    out_test_graphs = to_numpy(out_test_graphs)

    test_dist_acc = compute_dist_bacc(
        out_test_graphs, gt_test_graphs, bidim_solution)
    draw_acc(test_dist_acc, log_path)

    stages = get_stages(in_test_graphs, gt_test_graphs, out_test_graphs)
    draw_revertion(stages["steady"], stages["transient"], log_path)
