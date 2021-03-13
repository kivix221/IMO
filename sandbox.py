import numpy as np
from greedy_cycle import get_first_node, get_cycle_next_node


def basic_greedy_cycle(matrix: np.ndarray, node=0) -> list:
    not_been = list(range(len(matrix)))
    not_been.pop(node)
    cycle = [node, get_first_node(matrix, not_been, node)]
    not_been.remove(cycle[-1])

    while len(not_been) != 0:
        inx, ci = get_cycle_next_node(matrix, not_been, cycle)
        cycle.insert(ci + 1, not_been[inx])
        not_been.pop(inx)

    return cycle


def double_greedy_cycle(matrix: np.ndarray, node=0) -> (list, list):
    matrix[node, node] = 0
    # print(matrix[node])
    sec_node = np.argmax(matrix[node])
    matrix[node, node] = np.inf
    # print(sec_node)

    not_been = list(range(len(matrix)))
    not_been.pop(node)
    # print(not_been)
    not_been.remove(sec_node)
    cycle1 = [node, get_first_node(matrix, not_been, node)]
    not_been.remove(cycle1[-1])
    cycle2 = [sec_node, get_first_node(matrix, not_been, sec_node)]
    not_been.remove(cycle2[-1])

    while len(not_been) != 0:
        inx, ci = get_cycle_next_node(matrix, not_been, cycle1)
        cycle1.insert(ci + 1, not_been.pop(inx))

        if len(not_been) == 0:
            break
        inx, ci = get_cycle_next_node(matrix, not_been, cycle2)
        cycle2.insert(ci + 1, not_been.pop(inx))

    cycle1.append(node)
    cycle2.append(sec_node)
    return cycle1, cycle2


def double_greedy_cycle_seq(matrix: np.ndarray, node=0) -> (list, list):
    not_been = list(range(len(matrix)))
    not_been.pop(node)
    cycle1 = [node, get_first_node(matrix, not_been, node)]
    not_been.remove(cycle1[-1])
    half = len(matrix) // 2

    while len(not_been) > half:
        inx, ci = get_cycle_next_node(matrix, not_been, cycle1)
        cycle1.insert(ci + 1, not_been.pop(inx))

    sec_node = not_been.pop()  # możliwa randomizacja pobieranego wierzchołka
    cycle2 = [sec_node, get_first_node(matrix, not_been, sec_node)]
    while len(not_been) != 0:
        inx, ci = get_cycle_next_node(matrix, not_been, cycle2)
        cycle2.insert(ci + 1, not_been.pop(inx))

    cycle1.append(node)
    cycle2.append(sec_node)
    return cycle1, cycle2


# def double_regret_cycle_seq(matrix: np.ndarray, node=0) -> (list, list):
#     not_been = list(range(len(matrix)))
#     not_been.pop(node)
#     cycle1 = [node, get_first_node(matrix, not_been, node)]
#     not_been.remove(cycle1[-1])
#     half = len(matrix) // 2
#
#     while len(not_been) > half:
#         inx, ci = get_next_regret_node(matrix, not_been, cycle1)
#         cycle1.insert(ci + 1, not_been.pop(inx))
#
#     sec_node = not_been.pop()  # możliwa randomizacja pobieranego wierzchołka
#     cycle2 = [sec_node, get_first_node(matrix, not_been, sec_node)]
#     while len(not_been) != 0:
#         inx, ci = get_next_regret_node(matrix, not_been, cycle2)
#         cycle2.insert(ci + 1, not_been.pop(inx))
#
#     cycle1.append(node)
#     cycle2.append(sec_node)
#     return cycle1, cycle2


def get_next_regret_node2(matrix: np.ndarray, not_been: list, cycle: list) -> (int, int):
    mi, mv = 0, np.inf
    di, dv = 0, np.inf
    ci = -1
    max_regret, ci_regret, nb_regret = -np.inf, -1, -1
    for ix, vx in enumerate(not_been):
        for i in range(len(cycle)):
            dis = matrix[vx, cycle[i]] + matrix[cycle[(i + 1)%len(cycle)], vx] - matrix[cycle[i], cycle[(i + 1)%len(cycle)]]
            if dis < mv:
                ci = i
                di, dv = mi, mv
                mi, mv = ix, dis
            elif dis < dv:
                di, dv = ix, dis
        if mv - dv > max_regret:
            max_regret = mv-dv
            ci_regret = ci
            nb_regret = mi

    return nb_regret, ci_regret


def double_regret_cycle2(matrix: np.ndarray, node=0) -> (list, list):
    matrix[node, node] = 0
    # print(matrix[node])
    sec_node = np.argmax(matrix[node])
    matrix[node, node] = np.inf
    # print(sec_node)

    not_been = list(range(len(matrix)))
    not_been.pop(node)
    # print(not_been)
    not_been.remove(sec_node)
    cycle1 = [node, get_first_node(matrix, not_been, node)]
    not_been.remove(cycle1[-1])
    cycle2 = [sec_node, get_first_node(matrix, not_been, sec_node)]
    not_been.remove(cycle2[-1])

    while len(not_been) != 0:
        inx, ci = get_cycle_next_node(matrix, not_been, cycle1)
        cycle1.insert(ci + 1, not_been.pop(inx))
        if len(not_been) == 0:
            break
        inx, ci = get_cycle_next_node(matrix, not_been, cycle2)
        cycle2.insert(ci + 1, not_been.pop(inx))

    cycle1.append(node)
    cycle2.append(sec_node)
    return cycle1, cycle2