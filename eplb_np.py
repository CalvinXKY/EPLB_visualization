from typing import Tuple
import numpy as np

def balanced_packing(weight: list, num_packs: int) -> Tuple[list, list]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item (as nested list)
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers = len(weight)
    num_groups = len(weight[0])
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = [[j for j in range(num_groups)] for _ in range(num_layers)]
        rank_in_pack = [[0 for _ in range(num_groups)] for _ in range(num_layers)]
        return pack_index, rank_in_pack

    # 初始化输出列表
    pack_index = [[-1 for _ in range(num_groups)] for _ in range(num_layers)]
    rank_in_pack = [[-1 for _ in range(num_groups)] for _ in range(num_layers)]

    for i in range(num_layers):
        # 对每层的权重进行排序
        sorted_indices = np.argsort([-x for x in weight[i]])
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs

        for group in sorted_indices:
            # 找到当前最轻的包
            pack = min((j for j in range(num_packs) if pack_items[j] < groups_per_pack),
                      key=lambda x: pack_weights[x])
            assert pack_items[pack] < groups_per_pack
            pack_index[i][group] = pack
            rank_in_pack[i][group] = pack_items[pack]
            pack_weights[pack] += weight[i][group]
            pack_items[pack] += 1

    return pack_index, rank_in_pack

def replicate_experts(weight: list, num_phy: int) -> Tuple[list, list, list]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log] as nested list
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n = len(weight)
    num_log = len(weight[0])
    num_redundant = num_phy - num_log
    assert num_redundant >= 0

    # 初始化输出列表
    phy2log = [[i for i in range(num_phy)] for _ in range(n)]
    rank = [[0 for _ in range(num_phy)] for _ in range(n)]
    logcnt = [[1 for _ in range(num_log)] for _ in range(n)]

    for i in range(num_log, num_phy):
        for layer in range(n):
            # 计算每个专家的平均负载
            avg_loads = [weight[layer][j] / logcnt[layer][j] for j in range(num_log)]
            redundant_index = avg_loads.index(max(avg_loads))

            phy2log[layer][i] = redundant_index
            rank[layer][i] = logcnt[layer][redundant_index]
            logcnt[layer][redundant_index] += 1

    return phy2log, rank, logcnt

def inverse(perm: list) -> list:
    """将排列转换为其逆排列"""
    n_rows = len(perm)
    n_cols = len(perm[0])
    inv = [[-1 for _ in range(n_cols)] for _ in range(n_rows)]
    for i in range(n_rows):
        for j in range(n_cols):
            inv[i][perm[i][j]] = j
    return inv

def rebalance_experts_hierarchical(weight: list, num_physical_experts: int,
                      num_groups: int, num_nodes: int, num_gpus: int):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts] as nested list
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers = len(weight)
    num_logical_experts = len(weight[0])
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    # Step 1: pack groups to nodes
    # 计算每个组的总权重
    tokens_per_group = [[0 for _ in range(num_groups)] for _ in range(num_layers)]
    for i in range(num_layers):
        for j in range(num_groups):
            start_idx = j * group_size
            end_idx = start_idx + group_size
            tokens_per_group[i][j] = sum(weight[i][start_idx:end_idx])

    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)

    # 构建log2mlog映射
    log2mlog = [[0 for _ in range(num_logical_experts)] for _ in range(num_layers)]
    for i in range(num_layers):
        for g in range(num_groups):
            for k in range(group_size):
                idx = g * group_size + k
                node = group_pack_index[i][g]
                rank = group_rank_in_pack[i][g]
                log2mlog[i][idx] = (node * groups_per_node + rank) * group_size + k

    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    tokens_per_mlog = [[0 for _ in range(num_logical_experts // num_nodes)]
                      for _ in range(num_layers * num_nodes)]
    for i in range(num_layers):
        for j in range(num_logical_experts):
            node_idx = j // (num_logical_experts // num_nodes)
            local_idx = j % (num_logical_experts // num_nodes)
            layer_node_idx = i * num_nodes + node_idx
            tokens_per_mlog[layer_node_idx][local_idx] = weight[i][mlog2log[i][j]]

    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs
    tokens_per_phy = [[0 for _ in range(num_physical_experts // num_nodes)]
                     for _ in range(num_layers * num_nodes)]
    for i in range(len(tokens_per_mlog)):
        for j in range(len(phy2mlog[i])):
            mlog_idx = phy2mlog[i][j]
            tokens_per_phy[i][j] = tokens_per_mlog[i][mlog_idx] / mlogcnt[i][mlog_idx]

    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)

    # 构建最终映射
    phy2pphy = [[pack_index[i][j] * phy_experts_per_gpu + rank_in_pack[i][j]
                 for j in range(len(pack_index[i]))]
                for i in range(len(pack_index))]
    pphy2phy = inverse(phy2pphy)

    # 整合结果
    pphy2log = [[0 for _ in range(num_physical_experts)] for _ in range(num_layers)]
    pphyrank = [[0 for _ in range(num_physical_experts)] for _ in range(num_layers)]
    logcnt = [[0 for _ in range(num_logical_experts)] for _ in range(num_layers)]

    for i in range(num_layers):
        for j in range(num_physical_experts):
            node_idx = j // (num_physical_experts // num_nodes)
            local_idx = j % (num_physical_experts // num_nodes)
            layer_node_idx = i * num_nodes + node_idx

            mlog_idx = phy2mlog[layer_node_idx][pphy2phy[layer_node_idx][local_idx]]
            global_mlog_idx = node_idx * (num_logical_experts // num_nodes) + mlog_idx
            pphy2log[i][j] = mlog2log[i][global_mlog_idx]
            pphyrank[i][j] = phyrank[layer_node_idx][pphy2phy[layer_node_idx][local_idx]]

        for j in range(num_logical_experts):
            logcnt[i][j] = mlogcnt[i // num_nodes][log2mlog[i][j] % (num_logical_experts // num_nodes)]

    return pphy2log, pphyrank, logcnt

def rebalance_experts(weight: list, num_replicas: int, num_groups: int,
                      num_nodes: int, num_gpus: int) -> Tuple[list, list, list]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = replicate_experts(weight, num_replicas)

    maxlogcnt = max(max(cnt) for cnt in logcnt)
    num_layers = len(weight)
    num_logical_experts = len(weight[0])

    log2phy = [[[-1 for _ in range(maxlogcnt)]
                for _ in range(num_logical_experts)]
               for _ in range(num_layers)]

    for i in range(num_layers):
        for j in range(num_replicas):
            log_idx = phy2log[i][j]
            rank_idx = phyrank[i][j]
            log2phy[i][log_idx][rank_idx] = j

    return phy2log, log2phy, logcnt

__all__ = ['rebalance_experts']
