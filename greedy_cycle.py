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
            dis = matrix[vx, cycle[i]] + matrix[cycle[i + 1], vx]
            if dis < mv:
                ci = i
                mi = ix
                mv = dis
        dis = matrix[vx, cycle[0]] + matrix[cycle[len(cycle) - 1], vx]
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
    half = len(matrix)//2

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
