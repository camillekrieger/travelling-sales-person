import math
import random
from tsp_core import Tour, SolutionStats, Timer, score_tour, Solver, generate_network
from tsp_cuttree import CutTree
from math import inf
import copy
import numpy as np
import heapq

def random_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))

    while True:
        if timer.time_out():
            return stats

        tour = random.sample(list(range(len(edges))), len(edges))
        n_nodes_expanded += 1

        cost = score_tour(tour, edges)
        if math.isinf(cost):
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        if stats and cost > stats[-1].score:
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue

        stats.append(SolutionStats(
            tour=tour,
            score=cost,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))

    if not stats:
        return [SolutionStats(
            [],
            math.inf,
            timer.time(),
            1,
            n_nodes_expanded,
            n_nodes_pruned,
            cut_tree.n_leaves_cut(),
            cut_tree.fraction_leaves_covered()
        )]
    
def greedy_path(start, edges):
    temp_edges = copy.deepcopy(edges)
    path = [start]
    curr = temp_edges[start]
    valid = True
    while valid:
        valid = False
        for spot in curr:
            if spot != float('inf') and spot > 0:
                valid = True
                break
        min = float('inf')
        index = None
        for i in range(len(curr)):
            if curr[i] < min and curr[i] > 0 and i not in path:
                min = curr[i]
                index = i
            curr[i] = float('inf')
        if index != None:
            path.append(index)
            curr = temp_edges[index]
        else:
            return path
    return path

def greedy_tour(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))
    for i in range(len(edges)):
        if timer.time_out():
            return stats
        n_nodes_expanded += 1
        tour = greedy_path(i, edges)
        cost = score_tour(tour, edges)
        if len(tour) != len(edges[0]):
            continue
        if math.isinf(cost):
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue
        if stats and cost >= stats[-1].score:
            n_nodes_pruned += 1
            cut_tree.cut(tour)
            continue
        stats.append(SolutionStats(
            tour=tour,
            score=cost,
            time=timer.time(),
            max_queue_size=1,
            n_nodes_expanded=n_nodes_expanded,
            n_nodes_pruned=n_nodes_pruned,
            n_leaves_covered=cut_tree.n_leaves_cut(),
            fraction_leaves_covered=cut_tree.fraction_leaves_covered()
        ))
    if not stats:
        return [SolutionStats(
            [],
            math.inf,
            timer.time(),
            1,
            n_nodes_expanded,
            n_nodes_pruned,
            cut_tree.n_leaves_cut(),
            cut_tree.fraction_leaves_covered()
        )]
    return stats

def get_children(list, path):
    children = []
    for i in range(len(list)):
        if list[i] != float('inf') and list[i] > 0 and i not in path:
            children.append(i)
    return children

def dfs(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    stats = []
    paths = [[0]]
    while paths:
        if timer.time_out():
            return stats
        path = paths.pop()
        children = get_children(edges[path[-1]], path)
        for i in children:
            tour = []
            tour.extend(path)
            tour.append(i)
            if len(tour) == len(edges[0]):
                cost = score_tour(tour, edges)
                if math.isinf(cost):
                    continue
                if stats and cost >= stats[-1].score:
                    continue
                stats.append(SolutionStats(
                    tour=tour,
                    score=cost,
                    time=timer.time(),
                    max_queue_size=1,
                    n_nodes_expanded=0,
                    n_nodes_pruned=0,
                    n_leaves_covered=0,
                    fraction_leaves_covered=0
                ))
            else:
                paths.append(tour)
    return stats

def reduced_cost_matrix(edges, lower):
    lower_bound = lower
    matrix = np.array(edges)
    for i in range(len(matrix)):
        row = matrix[i]
        min_val = np.min(row)
        if min_val != float('inf') and min_val > 0:
            matrix[i] -= min_val
            lower_bound += min_val
    for j in range(len(matrix[0])):
        col = matrix[:, j]
        min_val = np.min(col)
        if min_val != float('inf') and min_val > 0:
            matrix[:, j] -= min_val
            lower_bound += min_val
    return matrix.tolist(), lower_bound


def make_infinity(tour, cost_matrix):
    from_node = tour[-2]
    to_node = tour[-1]
    matrix = [row[:] for row in cost_matrix]
    matrix[from_node] = [inf for _ in matrix[from_node]]
    for row in matrix:
        row[to_node] = inf 
    matrix[to_node][from_node] = inf
    matrix[to_node][tour[0]]  = inf
    return matrix

def branch_and_bound(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    for i in range(len(edges)):
        edges[i][i] = float('inf')
    greedy_sol = greedy_tour(edges, timer)
    BSSF = greedy_sol[0].score
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))
    cost_matrix, lower_bound = reduced_cost_matrix(edges, 0)
    paths = [(lower_bound, [0], cost_matrix)]
    heapq.heapify(paths)
    while paths:
        if timer.time_out():
            if stats:
                return stats
            else:
                stats.append(greedy_sol[0])
                return(stats)
        bound, path, curr_matrix = heapq.heappop(paths)
        if bound >= BSSF:
            n_nodes_pruned += 1
            continue
        children = get_children(edges[path[-1]], path)
        children_comparisons = {}
        for i in children:
            n_nodes_expanded += 1
            edge = [path[-1], i]
            tour = path + [i]
            child_matrix = make_infinity(tour, curr_matrix)
            temp_cost_matrix, reduced_bound = reduced_cost_matrix(child_matrix, 0)
            edge_cost = curr_matrix[edge[0]][edge[1]]
            total_bound = bound + edge_cost + reduced_bound
            if total_bound >= BSSF:
                n_nodes_pruned += 1
                cut_tree.cut(tour)
                continue
            children_comparisons[tuple(tour)] = [temp_cost_matrix, total_bound]
        for t, (matrix, bound) in children_comparisons.items():
            heapq.heappush(paths, (bound, list(t), matrix))
        if len(path) == len(edges):
            cost = score_tour(path, edges)
            if math.isinf(cost):
                n_nodes_pruned += 1
                cut_tree.cut(path)
                continue
            if cost >= BSSF:
                n_nodes_pruned += 1
                cut_tree.cut(path)
                continue
            stats.append(SolutionStats(
                tour=path,
                score=cost,
                time=timer.time(),
                max_queue_size=len(paths),
                n_nodes_expanded=n_nodes_expanded,
                n_nodes_pruned=n_nodes_pruned,
                n_leaves_covered=cut_tree.n_leaves_cut(),
                fraction_leaves_covered=cut_tree.fraction_leaves_covered()
            ))
            BSSF = cost
    return stats

def branch_and_bound_smart(edges: list[list[float]], timer: Timer) -> list[SolutionStats]:
    for i in range(len(edges)):
        edges[i][i] = float('inf')
    greedy_sol = greedy_tour(edges, timer)
    BSSF = greedy_sol[0].score
    stats = []
    n_nodes_expanded = 0
    n_nodes_pruned = 0
    cut_tree = CutTree(len(edges))
    cost_matrix, h0 = reduced_cost_matrix(edges, 0)
    g0 = 0
    f0 = g0 + h0
    use_pq = True
    switch_interval = 0.3 * timer.time_limit
    switch_count = 0
    pq = [(f0, -len([0]), g0, [0], cost_matrix)]
    dfs = []
    while pq or dfs:
        if timer.time_out():
            return stats if stats else [greedy_sol[0]]
        if switch_count >= switch_interval:
            use_pq = not use_pq
            switch_count = 0
            if use_pq:
                pq = [(g + reduced_cost_matrix(cm, 0)[1], -len(t), g, t, cm) for g, t, cm in dfs]
                heapq.heapify(pq)
                dfs = []
            else:
                dfs = [(g, t, cm) for _, _, g, t, cm in pq]
                pq = []
        if use_pq:
            f_curr, _, g_curr, path, curr_matrix = heapq.heappop(pq)
        else:
            g_curr, path, curr_matrix = dfs.pop()
            f_curr = g_curr + reduced_cost_matrix(curr_matrix, 0)[1]
        switch_count += 1
        if f_curr >= BSSF:
            n_nodes_pruned += 1
            continue
        if len(path) == len(edges):
            cost = score_tour(path, edges)
            if math.isinf(cost) or cost >= BSSF:
                n_nodes_pruned += 1
                cut_tree.cut(path)
                continue
            stats.append(SolutionStats(
                tour=path,
                score=cost,
                time=timer.time(),
                max_queue_size=len(pq) + len(dfs),
                n_nodes_expanded=n_nodes_expanded,
                n_nodes_pruned=n_nodes_pruned,
                n_leaves_covered=cut_tree.n_leaves_cut(),
                fraction_leaves_covered=cut_tree.fraction_leaves_covered()
            ))
            BSSF = cost
            continue
        children = get_children(edges[path[-1]], path)
        for i in children:
            n_nodes_expanded += 1
            edge = [path[-1], i]
            tour = path + [i]
            child_matrix = make_infinity(tour, curr_matrix)
            temp_matrix, h_child = reduced_cost_matrix(child_matrix, 0)
            edge_cost = curr_matrix[edge[0]][edge[1]]
            g_child = g_curr + edge_cost
            f_child = g_child + h_child
            if f_child >= BSSF:
                n_nodes_pruned += 1
                cut_tree.cut(tour)
                continue
            if use_pq:
                heapq.heappush(pq, (f_child, -len(tour), g_child, tour, temp_matrix))
            else:
                dfs.append((g_child, tour, temp_matrix))
    return stats