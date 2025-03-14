from eplb_np import rebalance_experts
from visual_tool import reshape_map, visualize_ep_inputs, visualize_4d_array


if __name__ == "__main__":
    weight = [[90, 132, 40, 61, 104, 165, 39, 4],
              [20, 107, 104, 64, 19, 197, 187, 157],
              [100, 107, 104, 64, 20, 197, 187, 157]]

    num_replicas = 24
    num_groups = 2
    num_nodes = 2
    num_gpus = 8

    phy2log, log2phy, logcnt = rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)
    np_phy2log = reshape_map(phy2log, num_nodes, num_gpus)
    print(phy2log)
    visualize_ep_inputs(weight)
    visualize_4d_array(np_phy2log)
