import numpy as np
from .greedy_cycle import get_first_node, get_cycle_next_node


def get_next_regret_node(matrix: np.ndarray, not_been: list, cycle1: list, cycle2: list) -> (int, int):
    """
    zwraca niewykorzystany jeszcze wierzchołek, jak również miejsce oraz cykl do którego ma zostać
    dodany, dla którego został wyliczony największy żal
    """
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
    """
    algorytm z metodą rozbudowy cyklu oparty na żalu zwracający dwie listy zawierające obliczone cykle
    """
    size2 = len(matrix) // 2
    size1 = size2 + len(matrix) % 2

    matrix = np.copy(matrix)

    matrix[node, node] = 0
    sec_node = np.argmax(matrix[node])
    matrix[node, node] = np.inf

    not_been = list(range(len(matrix)))
    not_been.pop(node)
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
