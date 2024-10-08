from time import time
from functools import partial
from os.path import isdir, join, normpath

import sonnet as snt
import numpy as np
import tensorflow as tf
import mlflow as mlf
from tqdm import tqdm
from ragn.utils_mlf import (
    load_pickle,
    log_params_from_omegaconf_dict,
    set_mlflow,
    save_pickle,
)
from sklearn.preprocessing import minmax_scale

# from memory_profiler import memory_usage

from ragn.ragn import RAGN
from ragn.utils import (
    # edge_binary_focal_crossentropy,
    edge_binary_focal_crossentropy_final,
    # get_signatures,
    get_bacc,
    get_f1,
    get_precision,
)
from ragn.data import batch_generator_from_files


def train_ragn(
    exp_name,
    load_runs,
    cfg_data,
    cfg_model,
    cfg_train,
    num_msg,
    seed,
    debug,
):
    if debug:
        tf.debugging.experimental.enable_dump_debug_info(
            "/tmp/tfdbg2_logdir",
            tensor_debug_mode="FULL_HEALTH",
            circular_buffer_size=-1,
        )
    run = set_mlflow(exp_name, fix_path=True, load_runs=load_runs)
    assert run is not None, "Run is None"
    start_epoch = run.data.metrics.get("epoch")
    seen_graphs = run.data.metrics.get("graph")
    start_epoch = int(start_epoch) if start_epoch is not None else 0
    seen_graphs = int(seen_graphs) if seen_graphs is not None else 0
    with mlf.start_run(run_id=run.info.run_id):
        log_params_from_omegaconf_dict(cfg_data)
        log_params_from_omegaconf_dict(cfg_train)
        log_params_from_omegaconf_dict(cfg_model)
        model = RAGN(**cfg_model)
        estimator = Train(
            model,
            start_epoch,
            seen_graphs,
            seed=seed,
            num_msg=num_msg,
            **cfg_data,
            **cfg_train
        )
        # with tf.profiler.experimental.Profile('profile_logdir'):
        estimator.train()
        # tf.profiler.experimental.server.start(6009)
        # tf.profiler.experimental.client.trace('grpc://localhost:6009',
        # '/nfs/tb_log', 2000)
        # estimator.train()


class Train(snt.Module):
    def __init__(
        self,
        model,
        start_epoch,
        seen_graphs,
        num_epochs,
        optimizer,
        init_lr,
        decay_steps,
        end_lr,
        power,
        cycle,
        train_data_size,
        train_batch_size,
        val_batch_size,
        train_data_path,
        val_data_path,
        num_msg,
        seed,
        msg_drop_ratio,
        weight,
        scale_features,
        delta_time_to_validate,
        compile,
    ):
        super(Train, self).__init__(name="Train")
        self._seed = seed
        self._start_epoch = start_epoch
        self._seen_graphs = seen_graphs
        np.random.seed(self._seed)
        tf.random.set_seed(self._seed)
        try:
            self._rs = load_pickle("random_state")
        except Exception:
            self._rs = np.random.RandomState(self._seed)
        self._num_msg = num_msg
        self._msg_drop_ratio = msg_drop_ratio
        self._best_acc = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._best_f1 = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._best_precision = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self._best_delta = tf.Variable(np.infty, trainable=False, dtype=tf.float32)
        self._delta_time_to_validate = delta_time_to_validate
        self._model = model
        self._train_data_size = train_data_size
        self._num_epochs = num_epochs
        self._train_batch_size = train_batch_size
        self._val_batch_size = val_batch_size
        # self._loss_fn = partial(edge_binary_focal_crossentropy, alpha=weight)
        self._loss_fn = partial(edge_binary_focal_crossentropy_final, alpha=weight)
        self._lr = tf.Variable(init_lr, trainable=False, dtype=tf.float32, name="lr")
        self._step = tf.Variable(0, trainable=False, dtype=tf.float32, name="tr_step")
        self._opt = snt.optimizers.__getattribute__(optimizer)(learning_rate=self._lr)  # type: ignore
        self._schedule_lr_fn = tf.keras.optimizers.schedules.PolynomialDecay(  # type: ignore
            init_lr, decay_steps, end_lr, power=power, cycle=cycle
        )
        self._train_data_path = train_data_path
        self._val_data_path = val_data_path
        self._batch_generator = partial(
            batch_generator_from_files,
            shuffle=True,
            random_state=self._rs,
            bidim_solution=False,
            scaler=minmax_scale if scale_features else None,
        )
        self.set_managers()
        if compile:
            # val_generator = self._batch_generator(
            #     self._val_data_path, self._val_batch_size
            # )
            # in_val_graphs, gt_val_graphs = next(val_generator)
            # in_signature, gt_signature = get_signatures(in_val_graphs, gt_val_graphs)
            # self._update_model_weights = tf.function(
            #     self.__update_model_weights,
            #     input_signature=[in_signature, gt_signature],
            # )
            # self._eval = tf.function(
            #     self.__eval,
            #     input_signature=[in_signature],
            # )
            self._update_model_weights = self.__update_model_weights
            self._eval = self.__eval
        else:
            self._update_model_weights = self.__update_model_weights
            self._eval = self.__eval

    def set_managers(self):
        artifact_path = mlf.get_artifact_uri()
        last_path = join(artifact_path, "last_ckpts")
        acc_path = join(artifact_path, "best_acc_ckpts")
        precision_path = join(artifact_path, "best_precision_ckpts")
        f1_path = join(artifact_path, "best_f1_ckpts")
        delta_path = join(artifact_path, "best_delta_ckpts")
        ckpt = tf.train.Checkpoint(
            step=self._step,
            optimizer=self._opt,
            model=self._model,
            best_acc=self._best_acc,
            best_f1=self._best_f1,
            best_precision=self._best_precision,
            best_delta=self._best_delta,
        )
        self._last_ckpt_manager = tf.train.CheckpointManager(
            ckpt, last_path, max_to_keep=1
        )
        self._best_acc_ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            acc_path,
            max_to_keep=3,
        )
        self._best_f1_ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            f1_path,
            max_to_keep=3,
        )
        self._best_precision_ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            precision_path,
            max_to_keep=3,
        )
        self._best_delta_ckpt_manager = tf.train.CheckpointManager(
            ckpt,
            delta_path,
            max_to_keep=3,
        )
        if isdir(normpath(last_path).split(":")[-1]):
            _ = ckpt.restore(self._last_ckpt_manager.latest_checkpoint)
            tqdm.write(
                "Restore training from {}\n\tEpoch : {}, "
                "Seen Graphs : {}, Best Acc : {}, Best F1 : {}"
                "Best Precision : {}, Best Delta : {}\n".format(
                    last_path,
                    self._best_acc.numpy(),
                    self._best_f1.numpy(),
                    self._best_precision.numpy(),
                    self._best_delta.numpy(),
                    self._start_epoch,
                    self._seen_graphs,
                )
            )

    def __update_model_weights(self, in_graphs, gt_graphs, num_msg):
        with tf.GradientTape() as tape:
            output_graphs = self._model(in_graphs, num_msg, True)
            loss = self._loss_fn(
                gt_graphs.edges,
                # output_graphs,
                output_graphs.edges,
                # min_num_msg=tf.cast(
                #     tf.math.ceil(self._msg_drop_ratio * num_msg), dtype=tf.int32
                # ),
            )
        gradients = tape.gradient(loss, self._model.trainable_variables)
        self._opt.apply(gradients, self._model.trainable_variables)
        self._step.assign_add(1)
        # self._step = self._step + 1
        self._lr.assign(self._schedule_lr_fn(self._step))
        # self._lr = self._schedule_lr_fn(self._step)
        # return output_graphs[-1], loss
        return output_graphs, loss

    def __eval(self, in_graphs, num_msg):
        output_graphs = self._model(in_graphs, num_msg, False)
        return output_graphs

    def __assess_val(self):
        f1 = []
        acc = []
        loss = []
        precision = []
        bar = tqdm(
            total=9430,
            desc="Processed Graphs for Validation",
            leave=False,
        )
        val_generator = self._batch_generator(
            self._val_data_path, self._val_batch_size, sample=0.2
        )
        for in_graphs, gt_graphs in val_generator:
            num_msg = (
                self._num_msg
                if isinstance(self._num_msg, int)
                else tf.cast(
                    tf.math.ceil(self._num_msg * tf.cast(max(in_graphs.n_node.numpy()), tf.float32)),  # type: ignore
                    tf.int32,
                )
            )
            out_graphs = self._eval(in_graphs, num_msg)
            acc.append(get_bacc(gt_graphs.edges, out_graphs.edges))  # type: ignore
            f1.append(get_f1(gt_graphs.edges, out_graphs.edges))  # type: ignore
            precision.append(get_precision(gt_graphs.edges, out_graphs.edges))  # type: ignore
            # acc.append(get_bacc(gt_graphs.edges, out_graphs[-1].edges))  # type: ignore
            # f1.append(get_f1(gt_graphs.edges, out_graphs[-1].edges))  # type: ignore
            # precision.append(get_precision(gt_graphs.edges, out_graphs[-1].edges))  # type: ignore
            loss.append(
                self._loss_fn(
                    gt_graphs.edges,  # type: ignore
                    out_graphs.edges,
                    #     gt_graphs.edges,  # type: ignore
                    #     out_graphs,
                    #     min_num_msg=tf.cast(
                    #         tf.math.ceil(self._msg_drop_ratio * num_msg), dtype=tf.int32
                    #     ),
                )
            )
            bar.update(in_graphs.n_node.shape[0])  # type: ignore
        return (
            tf.reduce_sum(acc) / len(acc),
            tf.reduce_sum(f1) / len(f1),
            tf.reduce_sum(precision) / len(precision),
            tf.reduce_sum(loss) / len(loss),
        )

    def train(self):
        # mem = []
        # start_time = time()
        # last_validation = start_time
        for epoch in range(self._start_epoch, self._num_epochs):
            epoch_bar = tqdm(
                total=self._train_data_size,
                initial=self._seen_graphs,
                desc="Processed Graphs",
                leave=False,
            )
            epoch_bar.set_postfix(epoch="{} / {}".format(epoch, self._num_epochs))
            train_generator = self._batch_generator(
                self._train_data_path,
                self._train_batch_size,
                seen_graphs=self._seen_graphs,
            )
            for in_graphs, gt_graphs in train_generator:
                num_msg = (
                    self._num_msg
                    if isinstance(self._num_msg, int)
                    else int(np.ceil(self._num_msg * max(in_graphs.n_node.numpy())))  # type: ignore
                )
                tr_out_graphs, tr_loss = self._update_model_weights(
                    in_graphs, gt_graphs, num_msg
                )
                self._seen_graphs += in_graphs.n_node.shape[0]  # type: ignore
                # mem.append(memory_usage((self._update_model_weights, (in_graphs, gt_graphs, num_msg)))[-1])
                # delta_time = time() - last_validation
                # if delta_time >= self._delta_time_to_validate:
                if (self._seen_graphs % (self._train_data_size // 4)) == 0 or (
                    self._seen_graphs == self._train_data_size
                ):
                    # last_validation = time()
                    tqdm.write("Validation Skip")
                    tr_acc = get_bacc(gt_graphs.edges, tr_out_graphs.edges)  # type: ignore
                    tr_f1 = get_f1(gt_graphs.edges, tr_out_graphs.edges)  # type: ignore
                    tr_precision = get_precision(gt_graphs.edges, tr_out_graphs.edges)  # type: ignore
                    val_acc, val_f1, val_precision, val_loss = self.__assess_val()
                    delta = tf.abs(tr_loss - val_loss)
                    save_pickle("random_state", self._rs)
                    mlf.log_metric("graph", self._seen_graphs)
                    mlf.log_metric("epoch", epoch)
                    mlf.log_metrics(
                        {  # type: ignore
                            "tr loss": tr_loss.numpy(),
                            "tr f1": tr_f1.numpy(),  # type: ignore
                            "tr bacc": tr_acc.numpy(),  # type: ignore
                            "tr precision": tr_precision.numpy(),  # type: ignore
                            "val loss": val_loss.numpy(),
                            "val f1": val_f1.numpy(),
                            "val bacc": val_acc.numpy(),
                            "val precision": val_precision.numpy(),
                            "delta": delta.numpy(),
                            "lr": self._lr.numpy(),
                        }
                    )
                    self._last_ckpt_manager.save()
                    if self._best_acc <= val_acc:
                        self._best_acc_ckpt_manager.save()
                        self._best_acc.assign(val_acc)
                    if self._best_f1 <= val_f1:
                        self._best_f1_ckpt_manager.save()
                        self._best_f1.assign(val_f1)
                    if self._best_precision <= val_precision:
                        self._best_precision_ckpt_manager.save()
                        self._best_precision.assign(val_precision)
                    if self._best_delta >= delta:
                        self._best_delta_ckpt_manager.save()
                        self._best_delta.assign(delta)
                    epoch_bar.set_postfix(
                        epoch="{} / {}".format(epoch, self._num_epochs),
                        tr_loss="{:.3f}".format(tr_loss.numpy()),
                        val_loss="{:.3f}".format(val_loss.numpy()),
                        # best_acc="{:.3f}".format(self._best_acc.numpy()),
                        # best_f1="{:.3f}".format(self._best_f1.numpy()),
                        best_precision="{:.3f}".format(self._best_precision.numpy()),
                        # best_delta="{:.4f}".format(self._best_delta.numpy()),
                    )
                epoch_bar.update(in_graphs.n_node.shape[0])  # type: ignore
                # del tr_loss
                # del tr_out_graphs
                # if self._seen_graphs >= 30:
                #     break
            epoch_bar.close()
            self._seen_graphs = 0
            # break

        # import pandas as pd
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        #
        # mem = pd.DataFrame({"step": range(1, len(mem) + 1), "memory_comsumption" : mem})
        # print(mem)
        # print(mem.diff())
        # sns.lineplot(mem, x="step", y="memory_comsumption")
        # plt.savefig("total_memory.pdf")
        # sns.lineplot(mem.diff(), x="step", y="memory_comsumption")
        # plt.savefig("diff_memory.pdf")
