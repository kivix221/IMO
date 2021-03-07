import numpy as np
import sys


def get_first_node(matrix: np.ndarray, not_been: list, node=0) -> int:
    i, m = -1, sys.maxsize
    for v in not_been:
        if matrix[v, node] < m:
            m = matrix[v, node]
            i = v
    return i


def get_cycle_next_node(matrix: np.ndarray, not_been: list, cycle: list) -> (int, int):
    mi, mv = 0, sys.maxsize
    ci = -1
    for ix, vx in enumerate(not_been):
        for i in range(len(cycle) - 1):
            dis = matrix[vx, cycle[i]] + matrix[cycle[i + 1], vx] - matrix[cycle[i], cycle[i + 1]]
            if dis < mv:
                ci = i
                mi = ix
                mv = dis
        dis = matrix[vx, cycle[0]] + matrix[cycle[len(cycle) - 1], vx] - matrix[cycle[0], cycle[len(cycle) - 1]]
        if dis < mv:
            ci = len(cycle) - 1
            mi = ix
            mv = dis
    return mi, ci


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


def get_next_regret_node(matrix: np.ndarray, not_been: list, cycle1: list, cycle2: list) -> (int, int):
    nbc1, d_c1, ic1 = -1, np.inf, -1
    nbc2, d_c2, ic2 = -1, np.inf, -1
    max_regret, nb_regret, ci_regret, cycle_regret = -np.inf, -1, -1, False

    for nb_i, nb in enumerate(not_been):
        for c1_i, c1 in enumerate(cycle1):
            dis = matrix[c1, nb] + matrix[nb, cycle1[(c1_i + 1) % len(cycle1)]] - matrix[
                c1, cycle1[(c1_i + 1) % len(cycle1)]]
            if dis < d_c1:
                nbc1, d_c1, ic1 = nb_i, dis, c1_i

        for c2_i, c2 in enumerate(cycle2):
            dis = matrix[c2, nb] + matrix[nb, cycle2[(c2_i + 1) % len(cycle2)]] - matrix[
                c2, cycle2[(c2_i + 1) % len(cycle2)]]
            if dis < d_c2:
                nbc2, d_c2, ic2 = nb_i, dis, c2_i

        regret = -abs(d_c1 - d_c2)
        if regret > max_regret:
            max_regret = regret
            cycle_regret = d_c1 > d_c2
            if not cycle_regret:
                nb_regret, ci_regret = nbc1, ic1
            else:
                nb_regret, ci_regret = nbc2, ic2

    return nb_regret, ci_regret, cycle_regret


def double_regret_cycle(matrix: np.ndarray, node=0) -> (list, list):
    size2 = len(matrix) // 2
    size1 = size2 + len(matrix) % 2

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

    while len(cycle1) < size1 and len(cycle2) < size2:
        nb_i, c_i, cyc = get_next_regret_node(matrix, not_been, cycle1, cycle2)
        if not cyc:
            cycle1.insert(c_i + 1, not_been.pop(nb_i))
        else:
            cycle2.insert(c_i + 1, not_been.pop(nb_i))

    while len(cycle1) < size1:
        inx, ci = get_cycle_next_node(matrix, not_been, cycle1)
        cycle1.insert(ci + 1, not_been.pop(inx))

    while len(cycle2) < size2:
        inx, ci = get_cycle_next_node(matrix, not_been, cycle2)
        cycle2.insert(ci + 1, not_been.pop(inx))

    cycle1.append(node)
    cycle2.append(sec_node)
    return cycle1, cycle2


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


def get_next_node(matrix: np.ndarray, not_been: list, cycle1: list, cycle2: list) -> (int, int):
    nbc1, d_c1, ic1 = -1, np.inf, -1
    nbc2, d_c2, ic2 = -1, np.inf, -1
    max_regret, nb_regret, ci_regret, cycle_regret = -np.inf, -1, -1, False

    for c1_i, c1 in enumerate(cycle1):
        for nb_i, nb in enumerate(not_been):
            dis = matrix[c1, nb] + matrix[nb, cycle1[(c1_i + 1) % len(cycle1)]] - matrix[
                c1, cycle1[(c1_i + 1) % len(cycle1)]]
            if dis < d_c1:
                nbc1, d_c1, ic1 = nb_i, dis, c1_i
    for c2_i, c2 in enumerate(cycle2):
        for nb_i, nb in enumerate(not_been):
            dis = matrix[c2, nb] + matrix[nb, cycle2[(c2_i + 1) % len(cycle2)]] - matrix[
                c2, cycle2[(c2_i + 1) % len(cycle2)]]
            if dis < d_c2:
                nbc2, d_c2, ic2 = nb_i, dis, c2_i

    if d_c1 <= d_c2:
        return nbc1, ic1, False
    else:
        return nbc2, ic2, True
    # return nb_regret, ci_regret, cycle_regret


def double_greedy_cycle1(matrix: np.ndarray, node=0) -> (list, list):
    size2 = len(matrix) // 2
    size1 = size2 + len(matrix) % 2

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

    while len(cycle1) < size1 and len(cycle2) < size2:
        nb_i, c_i, cyc = get_next_node(matrix, not_been, cycle1, cycle2)
        if not cyc:
            cycle1.insert(c_i + 1, not_been.pop(nb_i))
        else:
            cycle2.insert(c_i + 1, not_been.pop(nb_i))

    while len(cycle1) < size1:
        inx, ci = get_cycle_next_node(matrix, not_been, cycle1)
        cycle1.insert(ci + 1, not_been.pop(inx))

    while len(cycle2) < size2:
        inx, ci = get_cycle_next_node(matrix, not_been, cycle2)
        cycle2.insert(ci + 1, not_been.pop(inx))

    cycle1.append(node)
    cycle2.append(sec_node)
    return cycle1, cycle2
