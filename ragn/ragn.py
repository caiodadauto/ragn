import sonnet as snt
import tensorflow as tf
from functools import partial

from graph_nets import blocks
from graph_nets import modules
from graph_nets.blocks import (
    _validate_graph,
    broadcast_receiver_nodes_to_edges,
    broadcast_sender_nodes_to_edges,
)

from graph_nets import utils_tf
from ragn.utils import concat, unsorted_segment_softmax
from graph_nets.graphs import EDGES, SENDERS, RECEIVERS


class NlLrMLP(snt.Module):
    def __init__(
        self,
        conf,
        create_scale=False,
        create_offset=False,
        dropout_rate=0.25,
        name="NlLrMLP",
    ):
        super(NlLrMLP, self).__init__(name=name)
        self._norm_layers = []
        self._linear_layers = []
        self._dropout_rate = 0.25
        for hidden_size in conf:
            self._linear_layers.append(snt.Linear(hidden_size))
            self._norm_layers.append(snt.LayerNorm(-1, create_scale, create_offset))

    def __call__(self, inputs, is_training):
        outputs_op = inputs
        for linear, norm in zip(self._linear_layers, self._norm_layers):
            outputs_op = tf.nn.dropout(outputs_op, self._dropout_rate)
            outputs_op = linear(outputs_op)
            outputs_op = norm(outputs_op)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=0.2)
        return outputs_op


class NlLrLSTM(snt.Module):
    def __init__(
        self,
        conf,
        create_scale=False,
        create_offset=False,
        recurrent_dropout=0.25,
        name="NlLrLSTM",
    ):
        super(NlLrLSTM, self).__init__(name=name)
        self._hidden_size = conf[0]
        train_lstm = []
        test_lstm = []
        for _ in range(conf[1]):
            dropout_lstm, lstm = snt.lstm_with_recurrent_dropout(
                self._hidden_size, dropout=recurrent_dropout
            )
            test_lstm.append(lstm)
            train_lstm.append(dropout_lstm)
        self._test_lstm = snt.DeepRNN(test_lstm)
        self._train_lstm = snt.DeepRNN(train_lstm)
        self._norm = snt.LayerNorm(
            -1,
            create_scale=create_scale,
            create_offset=create_offset,
        )

    def initial_state(self, batch_size, is_training):
        if is_training:
            return self._train_lstm.initial_state(batch_size)
        else:
            init_states = self._test_lstm.initial_state(batch_size)
            return (init_states,)

    def __call__(self, inputs, prev_states, is_training):
        if is_training:
            outputs, next_states = self._train_lstm(inputs, prev_states)
        else:
            outputs, next_states = self._test_lstm(inputs, prev_states[0])
            next_states = (next_states,)
        outputs = self._norm(outputs)
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2)
        return outputs, next_states


class LinkTransformer(snt.Module):
    def __init__(
        self,
        mlp_conf,
        enc_conf,
        num_heads,
        create_scale=False,
        create_offset=False,
        name="LinkTransformer",
    ):
        super(LinkTransformer, self).__init__(name=name)
        model_fn = partial(
            make_mlp_model,
            create_scale=create_scale,
            create_offset=create_offset,
        )
        self._multihead_models = []
        self._ratio = 1.0 / tf.math.sqrt(tf.cast(mlp_conf[-1], tf.float32))
        self._head_encoder = model_fn(conf=mlp_conf)
        self._norm = snt.LayerNorm(-1, create_scale, create_offset)
        for _ in range(num_heads):
            self._multihead_models.append(
                [
                    make_conv1d_model(enc_conf, create_scale, create_offset),
                    (
                        model_fn(conf=mlp_conf + [mlp_conf[-1] // 2]),
                        model_fn(conf=mlp_conf + [mlp_conf[-1] - mlp_conf[-1] // 2]),
                    ),
                    model_fn(conf=mlp_conf),
                ]
            )

    def __call__(
        self,
        destination,
        sender_features,
        edge_features,
        receiver_features,
        senders,
        n_node,
        **kwargs,
    ):
        # Query >  F(queries)
        # Key > concat(P(senders), Q(concat(edge, receivers)))
        # Value > W(concat(edge, receivers))
        multihead = []
        edge_receivers = tf.concat([edge_features, receiver_features], -1)
        for (
            query_model,
            key_model,
            value_model,
        ) in self._multihead_models:
            query = query_model(destination, **kwargs)
            key = tf.concat(
                [
                    key_model[0](sender_features, **kwargs),
                    key_model[1](edge_receivers, **kwargs),
                ],
                -1,
            )
            value = value_model(edge_receivers, **kwargs)
            att = tf.nn.selu(
                tf.reduce_sum(tf.multiply(key, query), -1, keepdims=True) / self._ratio
            )
            norm_att = unsorted_segment_softmax(att, senders, tf.reduce_sum(n_node))
            multihead.append(tf.multiply(norm_att, value))
        encoded_multihead = self._head_encoder(tf.concat(multihead, -1), **kwargs)
        encoded_residual_multihead = tf.add(edge_features, encoded_multihead)
        return self._norm(encoded_residual_multihead)


class ScalarConv1D(snt.Module):
    def __init__(
        self,
        enc_conf,
        create_scale=False,
        create_offset=False,
        name="ScalarConv1D",
    ):
        super(ScalarConv1D, self).__init__(name=name)
        self._pools = []
        self._convs = []
        for conv_conf, pool_conf in enc_conf[:-1]:
            self._pools.append(
                partial(
                    tf.nn.max_pool1d,
                    ksize=pool_conf[0],
                    strides=pool_conf[1],
                    padding=pool_conf[2],
                )
            )
            self._convs.append(
                snt.Conv1D(
                    output_channels=conv_conf[0],
                    kernel_shape=conv_conf[1],
                    stride=conv_conf[2],
                    padding=conv_conf[3],
                )
            )
        self._mlp = make_mlp_model(enc_conf[-1], create_scale, create_offset)

    def __call__(self, inputs, **kwargs):
        outputs = tf.expand_dims(inputs[:, :-1], axis=-1)
        edge_weights = tf.expand_dims(inputs[:, -1], axis=-1)
        for conv, pooling in zip(self._convs, self._pools):
            outputs = conv(outputs)
            outputs = pooling(outputs)
        outputs = tf.concat([snt.flatten(outputs), edge_weights], axis=-1)
        outputs = self._mlp(outputs, **kwargs)
        return outputs


class Conv1D(snt.Module):
    def __init__(
        self,
        enc_conf,
        create_scale=False,
        create_offset=False,
        name="Conv1D",
    ):
        super(Conv1D, self).__init__(name=name)
        self._pools = []
        self._convs = []
        for conv_conf, pool_conf in enc_conf[:-1]:
            self._pools.append(
                partial(
                    tf.nn.max_pool1d,
                    ksize=pool_conf[0],
                    strides=pool_conf[1],
                    padding=pool_conf[2],
                )
            )
            self._convs.append(
                snt.Conv1D(
                    output_channels=conv_conf[0],
                    kernel_shape=conv_conf[1],
                    stride=conv_conf[2],
                    padding=conv_conf[3],
                )
            )
        self._mlp = make_mlp_model(enc_conf[-1], create_scale, create_offset)

    def __call__(self, inputs, **kwargs):
        outputs = tf.expand_dims(inputs, axis=-1)
        for conv, pooling in zip(self._convs, self._pools):
            outputs = conv(outputs)
            outputs = pooling(outputs)
        outputs = snt.flatten(outputs)
        outputs = self._mlp(outputs, **kwargs)
        return outputs


def make_lstm_model(conf, create_scale, create_offset):
    return NlLrLSTM(conf, create_scale, create_offset)


def make_link_transformer(mlp_conf, enc_conf, num_heads, create_scale, create_offset):
    return LinkTransformer(mlp_conf, enc_conf, num_heads, create_scale, create_offset)


def make_mlp_model(conf, create_scale, create_offset):
    return NlLrMLP(conf, create_scale, create_offset)


def make_scalar_conv1d_model(conf, create_scale, create_offset):
    return ScalarConv1D(conf, create_scale, create_offset)


def make_conv1d_model(conf, create_scale, create_offset):
    return Conv1D(conf, create_scale, create_offset)


class LocalLinkDecision(snt.Module):
    def __init__(
        self,
        mlp_conf,
        enc_conf,
        decision_conf,
        create_scale=False,
        create_offset=False,
        name="LocalLinkDecision",
    ):
        super(LocalLinkDecision, self).__init__(name=name)
        self._transformers = []
        self._link_decision = make_mlp_model(
            [mlp_conf[-1] // 2, mlp_conf[-1] // 4, 1], create_scale, create_offset
        )
        for _ in range(decision_conf[0]):
            self._transformers.append(
                make_link_transformer(
                    mlp_conf, enc_conf, decision_conf[1], create_scale, create_offset
                )
            )

    def __call__(self, graphs, **kwargs):
        destination = utils_tf.repeat(graphs.globals, graphs.n_edge)
        sender_features = tf.gather(graphs.nodes, graphs.senders)
        receiver_features = tf.gather(graphs.nodes, graphs.receivers)
        edge_features = graphs.edges
        senders = graphs.senders
        n_node = graphs.n_node
        for transformer in self._transformers:
            edge_features = transformer(
                destination,
                sender_features,
                edge_features,
                receiver_features,
                senders,
                n_node,
                **kwargs,
            )
        out_edges = unsorted_segment_softmax(
            self._link_decision(edge_features, **kwargs), senders, tf.reduce_sum(n_node)
        )
        return graphs.replace(edges=out_edges)


# class GraphIndependent(snt.Module):
#     def __init__(
#         self,
#         edge_model_fn,
#         node_model_fn,
#         name="MLPGraphIndependent",
#     ):
#         super(GraphIndependent, self).__init__(name=name)
#         self._network = modules.GraphIndependent(
#             edge_model_fn=edge_model_fn,
#             node_model_fn=node_model_fn,
#             global_model_fn=None,
#         )
#
#     def __call__(self, graphs, edge_model_kwargs=None, node_model_kwargs=None):
#         if edge_model_kwargs is None:
#             edge_model_kwargs = {}
#         if node_model_kwargs is None:
#             node_model_kwargs = {}
#         return self._network(
#             graphs,
#             edge_model_kwargs=edge_model_kwargs,
#             node_model_kwargs=node_model_kwargs,
#         )
#


class NeighborhoodAggregator(snt.Module):
    def __init__(self, reducer, to_sender=False, name="neighborhood_aggregator"):
        super(NeighborhoodAggregator, self).__init__(name=name)
        self._reducer = reducer
        self._to_sender = to_sender

    @snt.once
    def _create_bias(self, graphs):
        shape = graphs.nodes.shape[1:]
        self._bias = tf.Variable(tf.zeros(shape), trainable=True, dtype=tf.float32)

    def __call__(self, graphs):
        self._create_bias(graphs)
        _validate_graph(
            graphs,
            (
                EDGES,
                SENDERS,
                RECEIVERS,
            ),
            additional_message="when aggregating from node features.",
        )
        if graphs.nodes is not None and graphs.nodes.shape.as_list()[0] is not None:
            num_nodes = graphs.nodes.shape.as_list()[0]
        else:
            num_nodes = tf.reduce_sum(graphs.n_node)
        indices = graphs.senders if self._to_sender else graphs.receivers
        broadcast = (
            broadcast_receiver_nodes_to_edges
            if self._to_sender
            else broadcast_sender_nodes_to_edges
        )
        reduced_node_features = self._reducer(broadcast(graphs), indices, num_nodes)
        return reduced_node_features + self._bias


class GraphLSTM(snt.Module):
    def __init__(
        self,
        conf,
        create_scale=False,
        create_offset=False,
        reducer=tf.math.unsorted_segment_sum,
        name="GraphLSTM",
    ):
        super(GraphLSTM, self).__init__(name=name)
        lstm_model_fn = partial(make_lstm_model, conf, create_scale, create_scale)
        self._edge_block = blocks.RecurrentEdgeBlock(
            edge_recurrent_model_fn=lstm_model_fn,
            use_edges=True,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False,
        )
        self._node_block = blocks.RecurrentNodeBlock(
            node_recurrent_model_fn=lstm_model_fn,
            use_received_edges=True,
            use_sent_edges=False,
            use_nodes=True,
            use_globals=False,
            aggregator_model_fn=NeighborhoodAggregator,
        )
        self._edge_state = None
        self._node_state = None

    def reset_state(
        self,
        graphs,
        edge_state=None,
        node_state=None,
        **kwargs,
    ):
        node_batch_size = tf.reduce_sum(graphs.n_node)
        edge_batch_size = tf.reduce_sum(graphs.n_edge)
        if edge_state is not None:
            self._edge_state = edge_state
        else:
            self._edge_state = self._edge_block.initial_state(
                batch_size=edge_batch_size, **kwargs
            )
        if node_state is not None:
            self._node_state = node_state
        else:
            self._node_state = self._node_block.initial_state(
                batch_size=node_batch_size, **kwargs
            )

    def __call__(self, graphs, edge_model_kwargs=None, node_model_kwargs=None):
        if edge_model_kwargs is None:
            edge_model_kwargs = {}
        if node_model_kwargs is None:
            node_model_kwargs = {}

        partial_graphs, next_edge_state = self._edge_block(
            graphs, self._edge_state, edge_model_kwargs
        )
        out_graphs, next_node_state = self._node_block(
            partial_graphs, self._node_state, node_model_kwargs
        )
        self._node_state = next_node_state
        self._edge_state = next_edge_state
        return out_graphs


class RAGN(snt.Module):
    def __init__(
        self,
        enc_conf,
        mlp_conf,
        rnn_conf,
        decision_conf,
        create_offset=False,
        create_scale=False,
        name="RAGN",
    ):
        super(RAGN, self).__init__(name=name)
        self._encoder = modules.GraphIndependent(
            edge_model_fn=partial(
                make_scalar_conv1d_model,
                conf=enc_conf,
                create_scale=create_scale,
                create_offset=create_offset,
            ),
            node_model_fn=partial(
                make_mlp_model,
                conf=mlp_conf,
                create_scale=create_scale,
                create_offset=create_offset,
            ),
            global_model_fn=None,
        )
        self._core = GraphLSTM(rnn_conf, create_scale, create_offset)
        self._lookup = LocalLinkDecision(
            mlp_conf,
            enc_conf,
            decision_conf,
            create_scale,
            create_offset,
        )

    def __call__(self, graphs, num_processing_steps, is_training):
        out_graphs = []
        kwargs = dict(is_training=is_training)
        init_latent = self._encoder(
            graphs, edge_model_kwargs=kwargs, node_model_kwargs=kwargs
        )
        latent = init_latent
        self._core.reset_state(graphs, **kwargs)
        for _ in range(num_processing_steps):
            core_input = concat([init_latent, latent], axis=1, use_globals=False)
            latent = self._core(
                core_input, edge_model_kwargs=kwargs, node_model_kwargs=kwargs
            )
            out_graphs.append(self._lookup(latent, is_training=is_training))
        return out_graphs
