from functools import partial

import sonnet as snt
import tensorflow as tf
from graph_nets import modules
from graph_nets import blocks
from graph_nets.blocks import (
    _validate_graph,
    broadcast_receiver_nodes_to_edges,
    broadcast_sender_nodes_to_edges,
)
from graph_nets.graphs import EDGES, SENDERS, RECEIVERS
from graph_nets import utils_tf

from ragn.utils import unsorted_segment_softmax, unsorted_segment_norm_attention_sum
from ragn.snt_modules import (
    make_lstm_model,
    make_mlp_model,
    make_scaled_attention,
    make_routing_query_key_value,
)


class NeighborhoodAggregator(snt.Module):
    def __init__(self, reducer, to_senders=False, name="neighborhood_aggregator"):
        super(NeighborhoodAggregator, self).__init__(name=name)
        self._reducer = reducer
        self._to_senders = to_senders

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
        indices = graphs.senders if self._to_senders else graphs.receivers
        broadcast = (
            broadcast_receiver_nodes_to_edges
            if self._to_senders
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
        lstm_model_fn = partial(make_lstm_model, conf, create_scale, create_offset)
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
            neighborhood_reducer=reducer,
            aggregator_model_fn=NeighborhoodAggregator,  # type: ignore
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
            graphs, self._edge_state, edge_model_kwargs  # type: ignore
        )
        out_graphs, next_node_state = self._node_block(
            partial_graphs, self._node_state, node_model_kwargs  # type: ignore
        )
        self._node_state = next_node_state
        self._edge_state = next_edge_state
        return out_graphs


class RoutingTransformer(snt.Module):
    def __init__(
        self,
        query_conf,
        key_conf,
        value_conf,
        node_mlp_conf,
        concat_mlp_conf,
        num_heads,
        name="routing_transformer",
    ):
        super(RoutingTransformer, self).__init__(name=name)
        self.dim_v = value_conf[-1]
        self._multihead_attention_models = []
        self._linear_attention = modules.GraphIndependent(
            edge_model_fn=partial(make_mlp_model, conf=concat_mlp_conf),
            node_model_fn=partial(make_mlp_model, conf=concat_mlp_conf),
            global_model_fn=None,
        )
        for _ in range(num_heads):
            attention_aux_model = modules.GraphNetwork(
                edge_model_fn=partial(
                    make_routing_query_key_value,
                    query_conf=query_conf,
                    key_conf=key_conf,
                    value_conf=value_conf,
                ),
                node_model_fn=lambda: lambda x: x,
                global_model_fn=lambda: lambda x: x,
                edge_block_opt=dict(
                    use_edges=True,
                    use_receiver_nodes=True,
                    use_sender_nodes=True,
                    use_globals=False,
                ),
                node_block_opt=dict(
                    use_received_edges=False,
                    use_sent_edges=True,
                    use_nodes=False,
                    use_globals=False,
                ),
                global_block_opt=dict(
                    use_edges=False, use_nodes=False, use_globals=True
                ),
                reducer=unsorted_segment_norm_attention_sum,
            )
            attention_model = modules.GraphNetwork(
                edge_model_fn=make_scaled_attention,
                node_model_fn=partial(make_mlp_model, conf=node_mlp_conf),
                global_model_fn=lambda: lambda x: x,
                edge_block_opt=dict(
                    use_edges=True,
                    use_receiver_nodes=False,
                    use_sender_nodes=True,
                    use_globals=False,
                ),
                node_block_opt=dict(
                    use_received_edges=False,
                    use_sent_edges=True,
                    use_nodes=False,
                    use_globals=False,
                ),
                global_block_opt=dict(
                    use_edges=False, use_nodes=False, use_globals=True
                ),
            )
            self._multihead_attention_models.append(
                (attention_aux_model, attention_model)
            )

    def __call__(self, graphs, enc_graphs, is_training):
        multihead_attention_graphs = []
        dim_attention = tf.reduce_sum(graphs.n_edge)
        total_num_node = tf.reduce_sum(graphs.n_node)
        idx_destinations = graphs.globals + (
            tf.math.cumsum(graphs.n_node) - graphs.n_node
        )
        sum_edge_features_to_receivers = tf.math.unsorted_segment_sum(
            enc_graphs.edges, graphs.receivers, total_num_node
        )
        enc_destination_ips = utils_tf.repeat(
            tf.gather(sum_edge_features_to_receivers, idx_destinations), graphs.n_edge
        )
        enc_destination_prefix = utils_tf.repeat(
            tf.gather(enc_graphs.nodes, idx_destinations), graphs.n_edge
        )
        enc_destinations = tf.concat(
            [
                tf.gather(graphs.nodes, graphs.senders),
                graphs.edges + enc_destination_ips,
                enc_destination_prefix,
            ],
            axis=-1,
        )
        for attention_aux_model, attention_model in self._multihead_attention_models:
            multihead_attention_graphs.append(
                attention_model(
                    attention_aux_model(
                        graphs,
                        edge_model_kwargs={
                            "enc_destinations": enc_destinations,
                            "is_training": is_training,
                        },
                        reducer_kwargs={"dim_attention": dim_attention},
                    ),
                    edge_model_kwargs={
                        "dim_attention": dim_attention,
                        "dim_v": self.dim_v,
                    },
                    node_model_kwargs={"is_training": is_training},
                )
            )
        attention_graphs = self._linear_attention(
            utils_tf.concat(multihead_attention_graphs, use_globals=False, axis=-1),
            edge_model_kwargs={"is_training": is_training},
            node_model_kwargs={"is_training": is_training},
        )
        return attention_graphs


class LinkDecision(snt.Module):
    def __init__(
        self,
        link_decision_conf,
        create_scale=False,
        create_offset=False,
        name="LinkDecision",
    ):
        super(LinkDecision, self).__init__(name=name)
        num_layers = link_decision_conf[0]
        edge_output_conf = link_decision_conf[1]
        # node_output_conf = link_decision_conf[2]
        mlp_conf, concat_mlp_conf, num_heads = link_decision_conf[2:]
        self._routing_transformers = []
        for _ in range(num_layers):
            self._routing_transformers.append(
                (
                    RoutingTransformer(
                        mlp_conf,
                        mlp_conf,
                        mlp_conf,
                        mlp_conf,
                        concat_mlp_conf,
                        num_heads,
                    ),
                    snt.LayerNorm(
                        -1, create_scale=create_scale, create_offset=create_offset
                    ),
                )
            )
        self._linear = modules.GraphIndependent(
            edge_model_fn=partial(make_mlp_model, conf=edge_output_conf),
            # node_model_fn=partial(make_mlp_model, conf=node_output_conf),
            node_model_fn=None,
            global_model_fn=None,
        )

    def __call__(self, graphs, enc_graphs, is_training):
        outputs = graphs
        for transformer, norm_layer in self._routing_transformers:
            _outputs = transformer(outputs, enc_graphs, is_training)
            outputs = outputs.replace(
                nodes=norm_layer(tf.add(outputs.nodes, _outputs.nodes)),
                edges=norm_layer(tf.add(outputs.edges, _outputs.edges)),
            )
        outputs = self._linear(outputs, edge_model_kwargs={"is_training": is_training})
        return outputs.replace(  # type:ignore
            edges=unsorted_segment_softmax(
                outputs.edges,  # type:ignore
                outputs.senders,  # type:ignore
                tf.reduce_sum(outputs.n_node),  # type:ignore
            )
        )
