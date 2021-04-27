import numpy as np

from graph_nets import utils_np
from sklearn.metrics import balanced_accuracy_score


def compute_accuracy(target, output, distribution=False):
    acc_all = []
    solved_all = []
    acc_true_all = []
    acc_false_all = []
    solved_true_all = []
    solved_false_all = []

    tg_dict = utils_np.graphs_tuple_to_data_dicts(target)
    out_dict = utils_np.graphs_tuple_to_data_dicts(output)
    for tg_graph, out_graph in zip(tg_dict, out_dict):
        expect = np.argmax(tg_graph["edges"], axis=-1)
        predict = np.argmax(out_graph["edges"], axis=-1)
        true_mask = np.ma.masked_equal(expect, 1).mask
        false_mask = np.ma.masked_equal(expect, 0).mask

        acc = expect == predict
        acc_true = acc[true_mask]
        acc_false = acc[false_mask]

        solved = np.all(acc)
        solved_true = np.all(acc_true)
        solved_false = np.all(acc_false)

        # acc_all.append(np.mean(acc))
        acc_all.append(balanced_accuracy_score(expect, predict))
        acc_true_all.append(np.mean(acc_true))
        acc_false_all.append(np.mean(acc_false))

        solved_all.append(solved)
        solved_true_all.append(solved_true)
        solved_false_all.append(solved_false)
    acc_all = np.stack(acc_all)
    acc_true_all = np.stack(acc_true_all)
    acc_false_all = np.stack(acc_false_all)

    solved_all = np.stack(solved_all)
    solved_true_all = np.stack(solved_true_all)
    solved_false_all = np.stack(solved_false_all)
    if not distribution:
        acc_all = np.mean(acc_all)
        acc_true_all = np.mean(acc_true_all)
        acc_false_all = np.mean(acc_false_all)

        solved_all = np.mean(solved_all)
        solved_true_all = np.mean(solved_true_all)
        solved_false_all = np.mean(solved_false_all)
    return (
        acc_all,
        solved_all,
        acc_true_all,
        solved_true_all,
        acc_false_all,
        solved_false_all,
    )


def get_generator_path_metrics(inputs, targets, outputs):
    num_attempts = 3
    out_dicts = utils_np.graphs_tuple_to_data_dicts(outputs)
    in_dicts = utils_np.graphs_tuple_to_data_dicts(inputs)
    tg_dicts = utils_np.graphs_tuple_to_data_dicts(targets)

    def softmax_prob(x):  # pylint: disable=redefined-outer-name
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    n_graphs = len(tg_dicts)
    for tg_graph, out_graph, in_graph, idx_graph in zip(
        tg_dicts, out_dicts, in_dicts, range(n_graphs)
    ):
        n_node = out_graph["n_node"]
        tg_graph_dist = tg_graph["nodes"][:, 0]
        tg_graph_hops = tg_graph["nodes"][:, 1]
        out_graph_dist = np.zeros_like(tg_graph_dist)
        out_graph_hops = np.zeros_like(tg_graph_dist)
        end_node = np.argwhere(tg_graph_dist == 0).reshape(1)[0]
        for node in range(n_node):
            hops = 0
            strength = 0
            start = node
            sender = None
            reachable = True
            path = np.zeros(n_node, dtype=int)
            while start != end_node:
                path[start] += 1
                start_edges_idx = np.argwhere(out_graph["senders"] == start).reshape(
                    -1,
                )
                receivers = out_graph["receivers"][start_edges_idx]
                start_edges = out_graph["edges"][start_edges_idx]
                edges_prob = softmax_prob(start_edges)
                routing_links = edges_prob[:, 0] < edges_prob[:, 1]

                if path[start] > num_attempts:
                    if end_node in receivers:
                        edge_forward_idx = np.argwhere(receivers == end_node).reshape(
                            -1,
                        )[0]
                        routing_links = np.ones_like(routing_links, dtype=bool)
                        sender = start
                        start = end_node
                    else:
                        reachable = False
                        break
                else:
                    if not np.any(routing_links):
                        routing_links = ~routing_links
                        edges_idx_sort = np.argsort(
                            edges_prob[:, 0] - edges_prob[:, 1]
                        )[::-1]
                    else:
                        edges_idx_sort = np.argsort(edges_prob[routing_links][:, 1])

                    if path[start] <= len(edges_idx_sort):
                        edge_forward_idx = edges_idx_sort[-path[start]]
                    else:
                        edge_forward_idx = edges_idx_sort[-1]

                    sender = start
                    start = receivers[routing_links][edge_forward_idx]
                hops += 1
                strength += in_graph["edges"][start_edges_idx][routing_links][
                    edge_forward_idx
                ][0]
            if reachable:
                out_graph_dist[node] = strength
                if tg_graph_dist[node] > strength:
                    print(tg_graph_dist[node])
                    print(strength)
                out_graph_hops[node] = hops
        out_graph_hops = np.delete(out_graph_hops, end_node)
        out_graph_dist = np.delete(out_graph_dist, end_node)
        tg_graph_hops = np.delete(tg_graph_hops, end_node)
        tg_graph_dist = np.delete(tg_graph_dist, end_node)
        idx_non_zero = np.flatnonzero(out_graph_hops)
        unreachable_p = 1 - idx_non_zero.size / out_graph_dist.size
        if idx_non_zero.size > 0:
            diff_dist = tg_graph_dist[idx_non_zero] / out_graph_dist[idx_non_zero]
            diff_hops = tg_graph_hops[idx_non_zero] / out_graph_hops[idx_non_zero]
            yield (diff_dist, diff_hops, unreachable_p)
        else:
            yield (None, None, unreachable_p)


def aggregator_path_metrics(inputs, targets, outputs, distribution=False):
    n_graphs = targets.n_node.size
    idx_graph = 0
    none_idx = []
    hist_hops = []
    hist_dist = []
    batch_max_dist_diff = np.zeros(n_graphs)
    batch_min_dist_diff = np.zeros(n_graphs)
    batch_avg_dist_diff = np.zeros(n_graphs)
    batch_max_hops_diff = np.zeros(n_graphs)
    batch_min_hops_diff = np.zeros(n_graphs)
    batch_avg_hops_diff = np.zeros(n_graphs)
    batch_unreachable_p = np.zeros(n_graphs)
    metrics_graph_generator = get_generator_path_metrics(inputs, targets, outputs)
    for diff_dist, diff_hops, unreachable_p in metrics_graph_generator:
        batch_unreachable_p[idx_graph] = unreachable_p

        if np.any(diff_dist == None):
            none_idx.append(idx_graph)
        else:
            batch_max_dist_diff[idx_graph] = np.max(diff_dist)
            batch_min_dist_diff[idx_graph] = np.min(diff_dist)
            batch_avg_dist_diff[idx_graph] = np.mean(diff_dist)
            batch_max_hops_diff[idx_graph] = np.max(diff_hops)
            batch_min_hops_diff[idx_graph] = np.min(diff_hops)
            batch_avg_hops_diff[idx_graph] = np.mean(diff_hops)
            if distribution:
                hist_hops.append(diff_hops)
                hist_dist.append(diff_dist)
        idx_graph += 1
    batch_max_dist_diff = np.delete(batch_max_dist_diff, none_idx)
    batch_min_dist_diff = np.delete(batch_min_dist_diff, none_idx)
    batch_avg_dist_diff = np.delete(batch_avg_dist_diff, none_idx)
    batch_max_hops_diff = np.delete(batch_max_hops_diff, none_idx)
    batch_min_hops_diff = np.delete(batch_min_hops_diff, none_idx)
    batch_avg_hops_diff = np.delete(batch_avg_hops_diff, none_idx)
    if not distribution:
        return dict(
            avg_batch_max_dist_diff=np.mean(batch_max_dist_diff)
            if batch_max_dist_diff.size
            else np.infty,
            avg_batch_min_dist_diff=np.mean(batch_min_dist_diff)
            if batch_min_dist_diff.size
            else np.infty,
            avg_batch_avg_dist_diff=np.mean(batch_avg_dist_diff)
            if batch_avg_dist_diff.size
            else np.infty,
            avg_batch_max_hops_diff=np.mean(batch_max_hops_diff)
            if batch_max_hops_diff.size
            else np.infty,
            avg_batch_min_hops_diff=np.mean(batch_min_hops_diff)
            if batch_min_hops_diff.size
            else np.infty,
            avg_batch_avg_hops_diff=np.mean(batch_avg_hops_diff)
            if batch_avg_hops_diff.size
            else np.infty,
            max_batch_unreachable_p=np.max(batch_unreachable_p),
            min_batch_unreachable_p=np.min(batch_unreachable_p),
            avg_batch_unreachable_p=np.mean(batch_unreachable_p),
        )
    else:
        return {
            "percentage of unreachable paths": batch_unreachable_p,
            "difference of hops": np.concatenate(hist_hops),
            "difference of strength": np.concatenate(hist_dist),
        }


class Router:
    def __init__(self, node, probs, edge_weights, receivers, steady=False):
        self._id = node
        self._probs = probs
        self._steady = steady
        self._receivers = receivers
        self._edge_weights = edge_weights.copy()
        self._own_weight = 0.

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

    def update_neighbor_weight(self, header):
        for node, weight in header.items():
            mask = self._receivers == node
            if np.any(mask):
                indices = np.arange(0, len(self._receivers), 1)
                node_idx = indices[mask]
                self._neighbor_weights[node_idx] = weight
    
    def get_next_node(self):
        valid_mask = self._neighbor_weights < self._own_weight
        if np.all(~valid_mask):
            new_own_weight = np.max(self._neighbor_weights) + 1.
            valid_mask = np.ones(self._neighbor_weights.shape[0], dtype=bool)
        else:
            new_own_weight = self._own_weight
        indices = np.argsort(self._data, order=["true_prob", "diff_prob"])
        sorted_valid_mask = valid_mask[indices]
        valid_indices = indices[sorted_valid_mask]
        next_node_idx = valid_indices[0]
        next_node = self._receivers[next_node_idx]
        if new_own_weight == self._own_weight:
            return next_node, self._edge_weights[next_node_idx], None
        else:
            self._own_weight = new_own_weight
            return next_node, self._edge_weights[next_node_idx], {self._id: new_own_weight}

def flow(node, target, edge_weights, receivers, prob_links, mask, routers=None):
    header = {}
    source = node
    total_hops = 0
    total_cost = 0
    routers = {} if routers is None else routers
    while node != target:
        if node not in routers:
            routers[node] = Router(
                node,
                prob_links[mask[node]],
                edge_weights[mask[node]].flatten(),
                receivers[mask[node]],
            )
        routers[node].update_neighbor_weight(header)
        # print("source:", node)
        # print("header", header)
        # print("own weight", routers[node]._own_weight)
        # print("neighbor weight:", routers[node]._neighbor_weights)
        # print("edges weight:", edge_weights[mask[node]])
        # print("probs:", prob_links[mask[node]])
        # print("receivers:", receivers[mask[node]])
        node, cost, header_update = routers[node].get_next_node()
        total_cost += cost
        total_hops += 1
        if header_update is not None:
            header.update(header_update)
        # print("to", target)
        # print("next", node)
        # print("header update", header_update)
        # print("total cost", total_cost)
        # print("total hops", total_hops)
        # print()
    return total_cost, total_hops, routers[source]


def _get_metrics(sources, djk_cost, djk_hops, target, edge_weights, receivers, prob_links, mask, routers=None):
    steady_routers = {}
    metrics = {"cost": [], "hops": []}
    for node in sources:
        if node != target:
            cost, hops, source_router = flow(node, target, edge_weights, receivers, prob_links, mask, routers=routers)
            metrics["cost"].append(djk_cost[node] / cost if cost > 0 else 0)
            metrics["hops"].append(djk_hops[node] / hops if hops > 0 else 0)
            steady_routers[node] = source_router
    print("AVG cost", np.array(metrics["cost"]).mean())
    return metrics, steady_routers


def _mask_neighbors(idx):
    unique_idx = np.unique(idx)
    mask = np.zeros((unique_idx.shape[0], idx.shape[0]), dtype=bool)
    for i in unique_idx:
        mask[i] = idx == i
    return mask

def reverse_link(graph, target, djk_cost, djk_hops, edge_weights, sources):
    prob_links = graph.edges
    receivers = graph.receivers
    senders = graph.senders
    if sources is None:
        sources = np.unique(senders)
    mask = _mask_neighbors(senders)
    stages = {}
    print("TRANSIENT")
    stages["transient"], steady_routers = _get_metrics(sources, djk_cost, djk_hops, target, edge_weights, receivers, prob_links, mask)
    print("STEADY")
    stages["steady"], _ = _get_metrics(sources, djk_cost, djk_hops, target, edge_weights, receivers, prob_links, mask, routers=steady_routers)
    print()
    print()
    return stages


def get_stages(values):
    all_stages = dict(
        transient={"cost": [], "hops": []}, steady={"cost": [], "hops": []}
    )
    input, target, output = values.values()
    for graph_idx in range(len(input.n_node)):
        print("Graph", graph_idx)
        output_graph = parse_edges_probs(utils_np.get_graph(output[-1], graph_idx))
        target_graph = utils_np.get_graph(target, graph_idx)
        end_node = np.argwhere(target_graph.nodes[:, 0] == 0).reshape(1)[0]
        djk_cost = target_graph.nodes[:, 0]
        djk_hops = target_graph.nodes[:, 1]
        edge_weights = utils_np.get_graph(input, graph_idx).edges
        stages = reverse_link(
            output_graph, end_node, djk_cost, djk_hops, edge_weights, sources=None
        )
        all_stages["transient"]["cost"] += stages["transient"]["cost"]
        all_stages["transient"]["hops"] += stages["transient"]["hops"]
        all_stages["steady"]["cost"] += stages["steady"]["cost"]
        all_stages["steady"]["hops"] += stages["steady"]["hops"]
    all_stages["transient"]["cost"] = np.array(all_stages["transient"]["cost"])
    all_stages["transient"]["hops"] = np.array(all_stages["transient"]["hops"])
    all_stages["steady"]["cost"] = np.array(all_stages["steady"]["cost"])
    all_stages["steady"]["hops"] = np.array(all_stages["steady"]["hops"])
    return all_stages


# TODO: Make the prob normalization based on the neighborhood
def parse_edges_probs(graph):
    def softmax_prob(x):
        e = np.exp(x)
        return e / np.sum(e, axis=-1, keepdims=True)

    edges = graph.edges
    return graph.replace(edges=softmax_prob(edges))
