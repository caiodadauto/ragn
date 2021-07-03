import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import minmax_scale

import pytop
from ragn.utils import (
    networkx_to_graph_tuple_generator,
    get_signatures,
    compute_dist_bacc,
    to_numpy,
)
from ragn.draw import draw_acc, draw_revertion
from ragn.policy import get_stages
from ragn.ragn import RAGN


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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


def set_environment(
    restore_from,
    enc_conf,
    mlp_conf,
    rnn_conf,
    decision_conf,
    create_offset,
    create_scale,
    log_path,
    scale,
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

    epoch = 0
    seen_graphs = 0
    log_dir = os.path.join(log_path, restore_from)
    best_dir = os.path.join(log_dir, "best_ckpts")
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
    enc_conf,
    mlp_conf,
    rnn_conf,
    decision_conf,
    create_offset,
    create_scale,
    debug=False,
    scale=False,
    input_fields=None,
    title="",
):
    def eval(in_graphs):
        output_graphs = model(in_graphs, n_msg, is_training=False)
        return output_graphs[-1]

    model, global_step, random_state, scaler = set_environment(
        restore_from,
        enc_conf,
        mlp_conf,
        rnn_conf,
        decision_conf,
        create_offset,
        create_scale,
        log_path,
        scale,
    )
    _, val_generator = init_generator(
        test_path, -1, scaler, random_state, input_fields=input_fields
    )
    in_test_graphs, gt_test_graphs, _ = next(val_generator)
    in_signarute, gt_signature = get_signatures(in_test_graphs, gt_test_graphs)
    if not debug:
        eval = tf.function(eval, input_signature=[in_signarute])
    out_test_graphs = eval(in_test_graphs)

    in_test_graphs = to_numpy(in_test_graphs)
    gt_test_graphs = to_numpy(gt_test_graphs)
    out_test_graphs = to_numpy(out_test_graphs)

    test_dist_acc = compute_dist_bacc(out_test_graphs, gt_test_graphs)
    draw_acc(test_dist_acc, log_path, title)

    # stages = get_stages(in_test_graphs, gt_test_graphs, out_test_graphs)
    # draw_revertion(stages["steady"], stages["transient"], log_path)
