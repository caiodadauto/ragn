from functools import partial

import sonnet as snt
from graph_nets import modules
from graph_nets.utils_tf import concat as gn_concat

from ragn.gn_modules import GraphLSTM, LinkDecision
from ragn.snt_modules import make_edge_conv2d_model, make_node_conv2d_model


class RAGN(snt.Module):
    def __init__(
        self,
        lstm_conf,
        edge_enc_conf,
        node_enc_conf,
        link_decision_conf,
        create_offset=False,
        create_scale=False,
        name="RAGN",
    ):
        super(RAGN, self).__init__(name=name)
        self._encoder = modules.GraphIndependent(
            edge_model_fn=partial(
                make_edge_conv2d_model,
                conf=edge_enc_conf,
                create_scale=create_scale,
                create_offset=create_offset,
            ),
            node_model_fn=partial(
                make_node_conv2d_model,
                conf=node_enc_conf,
                create_scale=create_scale,
                create_offset=create_offset,
            ),
            global_model_fn=None,
        )
        self._core = GraphLSTM(lstm_conf, create_scale, create_offset)
        self._link_decision = LinkDecision(
            link_decision_conf,
            create_scale=create_scale,
            create_offset=create_offset,
        )

    def __call__(self, graphs, num_msg, is_training):
        # out_graphs = []
        kwargs = dict(is_training=is_training)
        init_latent = self._encoder(
            graphs, edge_model_kwargs=kwargs, node_model_kwargs=kwargs
        )
        latent = init_latent
        self._core.reset_state(graphs, **kwargs)
        for _ in range(num_msg):  # type: ignore
            core_input = gn_concat([init_latent, latent], axis=1, use_globals=False)
            latent = self._core(
                core_input, edge_model_kwargs=kwargs, node_model_kwargs=kwargs
            )
        #     out_graphs.append(
        #         self._link_decision(latent, init_latent, is_training=is_training)
        #     )
        # return out_graphs
        return self._link_decision(latent, init_latent, is_training=is_training)
