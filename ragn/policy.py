import numpy as np
from tqdm import tqdm
from graph_nets import utils_np

from ragn.utils import parse_edges_bi_probs


PRINT = False


class Router:
    def __init__(self, node, probs, edge_weights, receivers, steady=False):
        self._id = node
        self._probs = probs
        self._steady = steady
        self._receivers = receivers
        self._edge_weights = edge_weights.copy()
        self._own_weight = 0.0

        true_prob = np.zeros(self._probs.shape[0])
        mask_true_labels = self._probs[:, 0] < self._probs[:, 1]
        true_prob[mask_true_labels] = -1 * self._probs[mask_true_labels][:, 1]
        diff_prob = -1 * (self._probs[:, 1] - self._probs[:, 0])
        self._neighbor_weights = np.zeros(probs.shape[0])
        self._data = np.array(
            list(zip(true_prob, diff_prob)),
            dtype=[
                ("true_prob", np.float32),
                ("diff_prob", np.float32),
            ],
        )

    def update_weight(self, node, weight):
        mask = self._receivers == node
        if np.any(mask):
            indices = np.arange(0, len(self._receivers), 1)
            node_idx = indices[mask]
            self._neighbor_weights[node_idx] = weight

    def broadcast_weight(self, routers):
        for receiver in self._receivers:
            routers[receiver].update_weight(self._id, self._own_weight)

    def get_next_node(self, routers):
        valid_mask = self._neighbor_weights < self._own_weight
        if np.all(~valid_mask):
            self._own_weight = np.max(self._neighbor_weights) + 1.0
            valid_mask = np.ones(self._neighbor_weights.shape[0], dtype=bool)
            self.broadcast_weight(routers)
        indices = np.argsort(self._data, order=["true_prob", "diff_prob"])
        sorted_valid_mask = valid_mask[indices]
        valid_indices = indices[sorted_valid_mask]
        next_node_idx = valid_indices[0]
        next_node = self._receivers[next_node_idx]
        return next_node, self._edge_weights[next_node_idx]


def flow(node, target, edge_weights, receivers, prob_links, mask, routers):
    routers = routers.copy()
    source = node
    total_hops = 0
    total_cost = 0
    while node != target:
        if PRINT:
            print("source:", node)
            print("own weight", routers[node]._own_weight)
            print("neighbor weight:", routers[node]._neighbor_weights)
            print("edges weight:", edge_weights[mask[node]])
            print("probs:", prob_links[mask[node]])
            print("receivers:", receivers[mask[node]])
        node, cost = routers[node].get_next_node(routers)
        total_cost += cost
        total_hops += 1
        if PRINT:
            print("to", target)
            print("next", node)
            print("total cost", total_cost)
            print("total hops", total_hops)
            print()
    return total_cost, total_hops, routers[source]


def _get_metrics(
    sources,
    djk_cost,
    djk_hops,
    target,
    edge_weights,
    receivers,
    prob_links,
    mask,
    routers,
    stage="Transient",
):
    steady_routers = {}
    metrics = {"cost": [], "hops": []}
    bar = tqdm(total=len(sources),
               desc="Processing {}".format(stage), leave=False)
    for node in sources:
        if node != target:
            cost, hops, source_router = flow(
                node, target, edge_weights, receivers, prob_links, mask, routers=routers
            )
            metrics["cost"].append(djk_cost[node] / cost if cost > 0 else 0)
            metrics["hops"].append(djk_hops[node] / hops if hops > 0 else 0)
            steady_routers[node] = source_router
            bar.update()
    bar.close()
    return metrics, steady_routers


def _mask_neighbors(idx):
    unique_idx = np.unique(idx)
    mask = np.zeros((unique_idx.shape[0], idx.shape[0]), dtype=bool)
    for i in unique_idx:
        mask[i] = idx == i
    return mask


def reverse_link(graph, target, djk_cost, djk_hops, edge_weights, sources):
    routers = {}
    prob_links = graph.edges
    receivers = graph.receivers
    senders = graph.senders
    mask = _mask_neighbors(senders)
    if sources is None:
        sources = np.unique(senders)
    for node in sources:
        routers[node] = Router(
            node,
            prob_links[mask[node]],
            edge_weights[mask[node]].flatten(),
            receivers[mask[node]],
        )

    stages = {}
    stages["transient"], steady_routers = _get_metrics(
        sources, djk_cost, djk_hops, target, edge_weights, receivers, prob_links, mask, routers
    )
    stages["steady"], _ = _get_metrics(
        sources,
        djk_cost,
        djk_hops,
        target,
        edge_weights,
        receivers,
        prob_links,
        mask,
        routers=steady_routers,
    )
    return stages


def get_stages(in_graphs, gt_graphs, pred_graphs):
    all_stages = dict(
        transient={"cost": [], "hops": []}, steady={"cost": [], "hops": []}
    )
    n_graphs = len(in_graphs.n_node)
    bar = tqdm(total=n_graphs, desc="Processed Graphs")
    for graph_idx in range(n_graphs):
        pred_graph = parse_edges_bi_probs(
            utils_np.get_graph(pred_graphs, graph_idx))
        gt_graph = utils_np.get_graph(gt_graphs, graph_idx)
        end_node = np.argwhere(gt_graph.nodes[:, 0] == 0).reshape(1)[0]
        djk_cost = gt_graph.nodes[:, 0]
        djk_hops = gt_graph.nodes[:, 1]
        edge_weights = utils_np.get_graph(in_graphs, graph_idx).edges
        stages = reverse_link(
            pred_graph, end_node, djk_cost, djk_hops, edge_weights, sources=None
        )
        all_stages["transient"]["cost"] += stages["transient"]["cost"]
        all_stages["transient"]["hops"] += stages["transient"]["hops"]
        all_stages["steady"]["cost"] += stages["steady"]["cost"]
        all_stages["steady"]["hops"] += stages["steady"]["hops"]
        bar.update()
        bar.set_postfix(cost_transient=np.array(stages["transient"]["cost"]).mean(
        ), cost_steady=np.array(stages["steady"]["cost"]).mean())
    bar.close()
    all_stages["transient"]["cost"] = np.array(all_stages["transient"]["cost"])
    all_stages["transient"]["hops"] = np.array(all_stages["transient"]["hops"])
    all_stages["steady"]["cost"] = np.array(all_stages["steady"]["cost"])
    all_stages["steady"]["hops"] = np.array(all_stages["steady"]["hops"])
    return all_stages
