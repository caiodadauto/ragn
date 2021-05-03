import tree
import numpy as np
import sonnet as snt
import tensorflow as tf
from functools import partial

from graph_nets import blocks
from graph_nets import modules
from graph_nets import utils_tf, utils_np
from graph_nets.graphs import NODES, EDGES, GLOBALS, SENDERS, RECEIVERS
from graph_nets.blocks import (
    _validate_graph,
    broadcast_receiver_nodes_to_edges,
    broadcast_sender_nodes_to_edges,
)


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
            "All graphs or no graphs must contain {} features.".format(field_name)
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

        output = input_graphs[0].replace(nodes=nodes, edges=edges, globals=globals_)
        if axis != 0:
            return output
        n_node_per_tuple = tf.stack([tf.reduce_sum(gr.n_node) for gr in input_graphs])
        n_edge_per_tuple = tf.stack([tf.reduce_sum(gr.n_edge) for gr in input_graphs])
        offsets = _compute_stacked_offsets(n_node_per_tuple, n_edge_per_tuple)
        n_node = tf.concat(
            [gr.n_node for gr in input_graphs], axis=0, name="concat_n_node"
        )
        n_edge = tf.concat(
            [gr.n_edge for gr in input_graphs], axis=0, name="concat_n_edge"
        )
        receivers = [gr.receivers for gr in input_graphs if gr.receivers is not None]
        receivers = receivers or None
        if receivers:
            receivers = tf.concat(receivers, axis, name="concat_receivers") + offsets
        senders = [gr.senders for gr in input_graphs if gr.senders is not None]
        senders = senders or None
        if senders:
            senders = tf.concat(senders, axis, name="concat_senders") + offsets
        return output.replace(
            receivers=receivers, senders=senders, n_node=n_node, n_edge=n_edge
        )


class LeakyReluMLP(snt.Module):
    def __init__(self, hidden_size, n_layers, name="LeakyReluMLP"):
        super(LeakyReluMLP, self).__init__(name=name)
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._linear_layers = []
        for _ in range(self._n_layers - 1):
            self._linear_layers.append(
                snt.Linear(int(np.floor(self._hidden_size * 0.7)))
            )
        self._linear_layers.append(snt.Linear(self._hidden_size))

    def __call__(self, inputs, is_training):
        outputs_op = inputs
        for linear in self._linear_layers:
            outputs_op = linear(outputs_op)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=0.2)
        return outputs_op


class LeakyReluNormMLP(snt.Module):
    def __init__(self, hidden_size, n_layers, name="LeakyReluNormMLP"):
        super(LeakyReluNormMLP, self).__init__(name=name)
        self._n_layers = n_layers
        self._hidden_size = hidden_size
        self._linear_layers = []
        self._bn_layers = []
        for _ in range(self._n_layers - 1):
            self._linear_layers.append(
                snt.Linear(int(np.floor(self._hidden_size * 0.7)))
            )
            self._bn_layers.append(snt.BatchNorm(create_offset=True, create_scale=True))
        self._linear_layers.append(snt.Linear(self._hidden_size))
        self._bn_layers.append(snt.BatchNorm(create_offset=True, create_scale=True))

    def __call__(self, inputs, is_training):
        outputs_op = inputs
        for linear, bn in zip(self._linear_layers, self._bn_layers):
            outputs_op = linear(outputs_op)
            outputs_op = bn(outputs_op, is_training=is_training, test_local_stats=True)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=0.2)
        return outputs_op


class LeakyReluNormLSTM(snt.Module):
    def __init__(
        self, hidden_size, depth, recurrent_dropout=0.25, name="LeakyReluNormLSTM"
    ):
        super(LeakyReluNormLSTM, self).__init__(name=name)
        self._hidden_size = hidden_size
        tr_lstm = []
        test_lstm = []
        for _ in range(depth):
            dropout_lstm, lstm = snt.lstm_with_recurrent_dropout(
                self._hidden_size, dropout=recurrent_dropout
            )
            test_lstm.append(lstm)
            tr_lstm.append(dropout_lstm)
        self.test_lstm = snt.DeepRNN(test_lstm)
        self.tr_lstm = snt.DeepRNN(tr_lstm)
        self._batch_norm = snt.BatchNorm(create_offset=True, create_scale=True)

    def initial_state(self, batch_size, is_training):
        if is_training:
            return self.tr_lstm.initial_state(batch_size)
        else:
            init_states = self.test_lstm.initial_state(batch_size)
            return (init_states,)

    def __call__(self, inputs, prev_states, is_training):
        if is_training:
            outputs, next_states = self.tr_lstm(inputs, prev_states)
        else:
            outputs, next_states = self.test_lstm(inputs, prev_states[0])
            next_states = (next_states,)

        outputs = self._batch_norm(
            outputs, is_training=is_training, test_local_stats=True
        )
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2)
        return outputs, next_states


def make_lstm_model(hidden_size, depth):
    return LeakyReluNormLSTM(hidden_size, depth)


def make_mlp_model(hidden_size, n_layers, model):
    return model(hidden_size, n_layers)

    def __init__(
        self,
        hidden_size=24,
        n_layers=5,
        model=LeakyReluNormMLP,
        name="MLPGraphIndependent",
    ):
        super(MLPGraphIndependent, self).__init__(name=name)
        self._network = modules.GraphIndependent(
            edge_model_fn=partial(
                make_mlp_model, hidden_size=hidden_size, n_layers=n_layers, model=model
            ),
            node_model_fn=partial(
                make_mlp_model, hidden_size=hidden_size, n_layers=n_layers, model=model
            ),
            global_model_fn=None,
        )


class BiLocalRoutingNetwork(snt.Module):
    def __init__(
        self,
        hidden_size=24,
        n_layers=5,
        model=LeakyReluMLP,
        n_heads=3,
        name="BiLocalRoutingNetwork",
    ):
        super(BiLocalRoutingNetwork, self).__init__(name=name)
        model_fn = partial(make_mlp_model, n_layers=n_layers, model=model)
        self._multihead_models = []
        self._routing_layer = snt.Linear(2)
        self._final_node_model = model_fn(hidden_size=hidden_size)
        self._final_query_model = model_fn(hidden_size=hidden_size)
        for _ in range(n_heads):
            self._multihead_models.append(
                [
                    model_fn(hidden_size=hidden_size),
                    model_fn(hidden_size=(hidden_size // 2)),
                    model_fn(hidden_size=(hidden_size - (hidden_size // 2))),
                    model_fn(hidden_size=8),
                ]
            )

    def _unsorted_segment_softmax(self, x, idx, n_idx):
        op1 = tf.exp(x)
        op2 = tf.math.unsorted_segment_sum(op1, idx, n_idx)
        op3 = tf.gather(op2, idx)
        op4 = tf.divide(op1, op3)
        return op4

    def __call__(self, graphs, **kwargs):
        queries = utils_tf.repeat(graphs.globals, graphs.n_edge)
        senders_feature = tf.gather(graphs.nodes, graphs.senders)
        receivers_feature = tf.gather(graphs.nodes, graphs.receivers)
        edge_rec_pair = tf.concat([graphs.edges, receivers_feature], -1)

        multihead_routing = []
        for (
            query_model,
            sender_model,
            edge_rec_model,
            multi_model,
        ) in self._multihead_models:
            enc_queries = query_model(queries, **kwargs)
            enc_senders = sender_model(senders_feature, **kwargs)
            enc_edge_rec = edge_rec_model(edge_rec_pair, **kwargs)
            att_op = tf.reduce_sum(
                tf.multiply(tf.concat([enc_senders, enc_edge_rec], -1), enc_queries),
                -1,
                keepdims=True,
            )
            attention_input = tf.math.sigmoid(
                att_op
            )  # tf.nn.leaky_relu(att_op, alpha=0.2)
            attentions = self._unsorted_segment_softmax(
                attention_input, graphs.senders, tf.reduce_sum(graphs.n_node)
            )

            multhead_op1 = multi_model(edge_rec_pair, **kwargs)
            multhead_op2 = tf.multiply(attentions, multhead_op1)
            # multhead_op3 = tf.math.unsorted_segment_sum(
            #     multhead_op2, graphs.senders, tf.reduce_sum(graphs.n_node)
            # )
            multihead_routing.append(multhead_op2)
        node_attention_feature = tf.concat(multihead_routing, -1)
        final_features = self._final_node_model(
            tf.concat(
                [tf.gather(node_attention_feature, graphs.senders), graphs.edges], -1
            ),
            **kwargs,
        )
        final_queries = self._final_query_model(queries, **kwargs)
        output_edges = self._routing_layer(tf.multiply(final_features, final_queries))
        return graphs.replace(edges=output_edges)


class OneLocalRoutingNetwork(snt.Module):
    def __init__(
        self,
        hidden_size=24,
        n_layers=5,
        model=LeakyReluMLP,
        n_heads=3,
        name="OneLocalRoutingNetwork",
    ):
        super(OneLocalRoutingNetwork, self).__init__(name=name)
        model_fn = partial(make_mlp_model, n_layers=n_layers, model=model)
        self._multihead_models = []
        self._routing_layer = snt.Linear(1)
        self._final_node_model = model_fn(hidden_size=hidden_size)
        self._final_query_model = model_fn(hidden_size=hidden_size)
        for _ in range(n_heads):
            self._multihead_models.append(
                [
                    model_fn(hidden_size=hidden_size),
                    model_fn(hidden_size=(hidden_size // 2)),
                    model_fn(hidden_size=(hidden_size - (hidden_size // 2))),
                    model_fn(hidden_size=8),
                ]
            )

    def _unsorted_segment_softmax(self, x, idx, n_idx):
        op1 = tf.exp(x)
        op2 = tf.math.unsorted_segment_sum(op1, idx, n_idx)
        op3 = tf.gather(op2, idx)
        op4 = tf.divide(op1, op3)
        return op4

    def __call__(self, graphs, **kwargs):
        queries = utils_tf.repeat(graphs.globals, graphs.n_edge)
        senders_feature = tf.gather(graphs.nodes, graphs.senders)
        receivers_feature = tf.gather(graphs.nodes, graphs.receivers)
        edge_rec_pair = tf.concat([graphs.edges, receivers_feature], -1)

        multihead_routing = []
        for (
            query_model,
            sender_model,
            edge_rec_model,
            multi_model,
        ) in self._multihead_models:
            enc_queries = query_model(queries, **kwargs)
            enc_senders = sender_model(senders_feature, **kwargs)
            enc_edge_rec = edge_rec_model(edge_rec_pair, **kwargs)
            att_op = tf.reduce_sum(
                tf.multiply(tf.concat([enc_senders, enc_edge_rec], -1), enc_queries),
                -1,
                keepdims=True,
            )
            attention_input = tf.math.sigmoid(att_op)
            attentions = self._unsorted_segment_softmax(
                attention_input, graphs.senders, tf.reduce_sum(graphs.n_node)
            )

            multhead_op1 = multi_model(edge_rec_pair, **kwargs)
            multhead_op2 = tf.multiply(attentions, multhead_op1)
            multhead_op3 = tf.math.unsorted_segment_sum(
                multhead_op2, graphs.senders, tf.reduce_sum(graphs.n_node)
            )
            multihead_routing.append(multhead_op3)
        node_attention_feature = tf.concat(multihead_routing, -1)
        final_features = self._final_node_model(
            tf.concat(
                [tf.gather(node_attention_feature, graphs.senders), graphs.edges], -1
            ),
            **kwargs,
        )
        final_queries = self._final_query_model(queries, **kwargs)
        output_edges = self._routing_layer(tf.multiply(final_features, final_queries))
        return graphs.replace(
            edges=self._unsorted_segment_softmax(
                tf.math.sigmoid(output_edges),
                graphs.senders,
                tf.reduce_sum(graphs.n_node),
            )
        )


class MLPGraphIndependent(snt.Module):
    def __init__(
        self,
        hidden_size=24,
        n_layers=5,
        model=LeakyReluNormMLP,
        name="MLPGraphIndependent",
    ):
        super(MLPGraphIndependent, self).__init__(name=name)
        self._network = modules.GraphIndependent(
            edge_model_fn=partial(
                make_mlp_model, hidden_size=hidden_size, n_layers=n_layers, model=model
            ),
            node_model_fn=partial(
                make_mlp_model, hidden_size=hidden_size, n_layers=n_layers, model=model
            ),
            global_model_fn=None,
        )

    def __call__(self, graphs, edge_model_kwargs=None, node_model_kwargs=None):
        if edge_model_kwargs is None:
            edge_model_kwargs = {}
        if node_model_kwargs is None:
            node_model_kwargs = {}
        return self._network(
            graphs,
            edge_model_kwargs=edge_model_kwargs,
            node_model_kwargs=node_model_kwargs,
        )


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


class GraphRecurrentNonLocalNetwork(snt.Module):
    def __init__(
        self,
        hidden_size=24,
        depth=2,
        reducer=tf.math.unsorted_segment_sum,
        name="GraphRecurrentNonLocalNetwork",
    ):
        super(GraphRecurrentNonLocalNetwork, self).__init__(name=name)
        recurrent_model_fn = partial(
            make_lstm_model, hidden_size=hidden_size, depth=depth
        )
        self._edge_block = blocks.RecurrentEdgeBlock(
            edge_recurrent_model_fn=recurrent_model_fn,
            use_edges=True,
            use_receiver_nodes=True,
            use_sender_nodes=True,
            use_globals=False,
        )
        self._node_block = blocks.RecurrentNodeBlock(
            node_recurrent_model_fn=recurrent_model_fn,
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
        hidden_size=24,
        n_layers=5,
        rnn_depth=2,
        n_heads=3,
        bidim=True,
        name="RAGN",
    ):
        super(RAGN, self).__init__(name=name)
        self._encoder = MLPGraphIndependent(hidden_size=hidden_size, n_layers=n_layers)
        self._core = GraphRecurrentNonLocalNetwork(
            hidden_size=hidden_size, depth=rnn_depth
        )
        if bidim:
            self._lookup = BiLocalRoutingNetwork(
                hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads
            )
        else:
            self._lookup = OneLocalRoutingNetwork(
                hidden_size=hidden_size, n_layers=n_layers, n_heads=n_heads
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
