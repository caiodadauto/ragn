import sonnet as snt
import tensorflow as tf


def make_mlp_model(conf):
    return MLP(conf)


def make_lstm_model(conf, create_scale, create_offset):
    return NormLSTM(conf, create_scale, create_offset)


def make_norm_mlp_model(conf, create_scale, create_offset):
    return NormMLP(conf, create_scale, create_offset)


def make_routing_query_key_value(query_conf, key_conf, value_conf):
    return RoutingQueryKeyValue(query_conf, key_conf, value_conf)


def make_scaled_attention():
    return ScaledAttention()


def make_edge_conv2d_model(conf, create_scale, create_offset):
    return EdgeConv2D(conf, create_scale, create_offset)


def make_node_conv2d_model(conf, create_scale, create_offset):
    return NodeConv2D(conf, create_scale, create_offset)


class NormLSTM(snt.Module):
    def __init__(
        self,
        conf,
        create_scale=False,
        create_offset=False,
        recurrent_dropout=0.35,
        name="NormLSTM",
    ):
        super(NormLSTM, self).__init__(name=name)
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
        # self._norm = snt.BatchNorm(create_scale, create_offset)
        self._norm = snt.LayerNorm(
            -1, create_scale=create_scale, create_offset=create_offset
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
        outputs = self._norm(outputs, is_training)  # type: ignore
        outputs = tf.nn.leaky_relu(outputs, alpha=0.2)
        return outputs, next_states


class MLP(snt.Module):
    def __init__(
        self,
        conf,
        dropout_rate=0.35,
        name="MLP",
    ):
        super(MLP, self).__init__(name=name)
        self._linear_layers = []
        self._dropout_rate = dropout_rate
        for hidden_size in conf:
            self._linear_layers.append(snt.Linear(hidden_size))

    def __call__(self, inputs, is_training):
        outputs_op = inputs
        for linear in self._linear_layers:
            if is_training:
                outputs_op = tf.nn.dropout(outputs_op, self._dropout_rate)
            outputs_op = linear(outputs_op)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=0.2)
        return outputs_op


class NormMLP(snt.Module):
    def __init__(
        self,
        conf,
        create_scale=False,
        create_offset=False,
        dropout_rate=0.35,
        name="NormMLP",
    ):
        super(NormMLP, self).__init__(name=name)
        self._norms = []
        self._linear_layers = []
        self._dropout_rate = dropout_rate
        for hidden_size in conf:
            self._linear_layers.append(snt.Linear(hidden_size))
            # self._norms.append(snt.BatchNorm(create_scale, create_offset))
            self._norms.append(
                snt.LayerNorm(
                    -1, create_scale=create_scale, create_offset=create_offset
                )
            )

    def __call__(self, inputs, is_training):
        outputs_op = inputs
        for linear, norm in zip(self._linear_layers, self._norms):
            if is_training:
                outputs_op = tf.nn.dropout(outputs_op, self._dropout_rate)
            outputs_op = linear(outputs_op)
            outputs_op = norm(outputs_op, is_training)
            outputs_op = tf.nn.leaky_relu(outputs_op, alpha=0.2)
        return outputs_op


class ScaledAttention(snt.Module):
    def __init__(self, name="scaled_attention"):
        super(ScaledAttention, self).__init__(name=name)

    def __call__(self, inputs, dim_attention, dim_v):
        alpha = inputs[:, :dim_attention]
        value = inputs[:, dim_attention : (dim_attention + dim_v)]
        norm = inputs[:, (dim_attention + dim_v) :]
        scaled_attention = tf.einsum("...ij,...jk->...ik", alpha / norm, value)
        return scaled_attention


class RoutingQueryKeyValue(snt.Module):
    def __init__(
        self,
        query_conf,
        key_conf,
        value_conf,
        name="routing_query_key_value",
    ):
        super(RoutingQueryKeyValue, self).__init__(name=name)
        self._query_model = MLP(query_conf)
        self._key_model = MLP(key_conf)
        self._value_model = MLP(value_conf)
        self._dim_key = tf.cast(key_conf[-1], tf.float32)

    def __call__(self, inputs, enc_destinations, is_training):
        query = self._query_model(enc_destinations, is_training)
        key = tf.multiply(
            self._key_model(inputs, is_training), 1.0 / tf.sqrt(self._dim_key)
        )
        value = self._value_model(inputs, is_training)
        key_t = tf.einsum("...ij->...ji", key)
        alpha = tf.math.exp(tf.einsum("...ij,...jk->...ik", query, key_t))
        return tf.concat([alpha, value], axis=-1)


class EdgeConv2D(snt.Module):
    def __init__(
        self,
        config,
        create_scale=False,
        create_offset=False,
        name="EdgeConv2D",
    ):
        super(EdgeConv2D, self).__init__(name=name)
        self._convs = []
        for conv_config in config[0]:
            self._convs.append(
                snt.Conv2D(
                    output_channels=conv_config[0],
                    kernel_shape=conv_config[1],
                    stride=conv_config[2],
                    padding=conv_config[3],
                )
            )
        self._mlp = make_norm_mlp_model(config[1], create_scale, create_offset)

    def __call__(self, inputs, **kwargs):
        attentions = inputs[:, -1]
        attention_inputs = tf.cast(inputs[:, :-1], tf.float32) * attentions
        attention_ips = tf.reshape(attention_inputs, shape=(-1, 4, 8, 1))
        outputs = attention_ips
        for conv in self._convs[0:2]:
            outputs = conv(outputs)
            outputs = outputs * tf.repeat(
                attention_ips, repeats=[outputs.shape[3]], axis=3
            )
        for conv in self._convs[2:]:
            outputs = conv(outputs)
        outputs = snt.flatten(outputs) * attentions
        outputs = self._mlp(outputs, **kwargs)
        return outputs


class NodeConv2D(snt.Module):
    def __init__(
        self,
        config,
        create_scale=False,
        create_offset=False,
        name="NormConv2D",
    ):
        super(NodeConv2D, self).__init__(name=name)
        self._convs = []
        for conv_config in config[0]:
            self._convs.append(
                snt.Conv2D(
                    output_channels=conv_config[0],
                    kernel_shape=conv_config[1],
                    stride=conv_config[2],
                    padding=conv_config[3],
                )
            )
        self._mlp = make_norm_mlp_model(config[1], create_scale, create_offset)

    def __call__(self, inputs, **kwargs):
        prefixes = tf.reshape(inputs, shape=(-1, 4, 8, 1))
        outputs = prefixes
        for conv in self._convs[0:2]:
            outputs = conv(outputs)
            outputs = outputs * tf.repeat(prefixes, repeats=[outputs.shape[3]], axis=3)
        for conv in self._convs[2:]:
            outputs = conv(outputs)
        outputs = self._mlp(outputs, **kwargs)
        return outputs
