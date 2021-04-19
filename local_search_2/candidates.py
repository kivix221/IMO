try:
    from ..utils import *
except Exception:
    from utils import *
import numpy as np


def calculate_candidates(dm: np.ndarray, n: int = 10):
    candidates = np.empty((len(dm), n), dtype=np.uint16)
    tmp_min = np.empty((n,), dtype=np.uint16)

    for i, d in enumerate(dm):
        tmp_min.fill(np.argmax(d))
        for j, v in enumerate(d):
            if v < d[tmp_min[-1]]:
                if v >= d[tmp_min[-2]]:
                    tmp_min[-1] = j
                else:
                    for m, t_min in enumerate(tmp_min):
                        if v < d[t_min]:
                            tmp_min[(m + 1):] = tmp_min[m:-1]
                            tmp_min[m] = j
                            break
        candidates[i] = tmp_min
    return candidates


def is_in_cycle(n: int, cycle1: np.ndarray, cycle2: np.ndarray):
    in_cycle = np.zeros((n,), dtype=np.int32)
    for i, c in enumerate(cycle1[:-1]):
        in_cycle[c] = i + 1
    for i, c in enumerate(cycle2[:-1]):
        in_cycle[c] = - i - 1
    return in_cycle


def update_in_cycle(in_cycle: np.ndarray, cycle: np.ndarray, mul: int):
    for i, v in enumerate(cycle[:-1]):
        in_cycle[v] = mul * (i + 1)
    return in_cycle


def steep_candidates(matrix, cycle1, cycle2, candidates, cycle_inx):
    search = True
    while search:
        search = False
        s, i, j = search_swap_in_candidates(matrix, cycle1[:-1], cycle2[:-1], candidates, cycle_inx)
        if i is not None or j is not None:
            search = True
            if s:
                i, j = min(i, j), max(i, j)
                cycle1[i + 1:j + 1] = cycle1[i + 1:j + 1][::-1]
                cycle_inx = update_in_cycle(cycle_inx, cycle1, 1)
            else:
                cycle1[i], cycle2[j] = cycle2[j], cycle1[i]
                cycle_inx[cycle1[i]] = i + 1
                cycle_inx[cycle2[j]] = -j - 1

    search = True
    while search:
        search = False
        s, i, j = search_swap_in_candidates(matrix, cycle2[:-1], cycle1[:-1], candidates, cycle_inx)
        if i is not None or j is not None:
            search = True
            if s:
                i, j = min(i, j), max(i, j)
                cycle2[i + 1:j + 1] = cycle2[i + 1:j + 1][::-1]
                cycle_inx = update_in_cycle(cycle_inx, cycle2, -1)
            else:
                cycle2[i], cycle1[j] = cycle1[j], cycle2[i]
                cycle_inx[cycle2[i]] = -i - 1
                cycle_inx[cycle1[j]] = j + 1

    cycle1[-1], cycle2[-1] = cycle1[-0], cycle2[0]
    return cycle1, cycle2


def search_swap_in_candidates(matrix, cycle1, cycle2, candidates, cycle_inx):
    best_i, best_j = None, None
    best_d = 0
    same_cycle = True
    for i, c1 in enumerate(cycle1):
        for candidate in candidates[c1]:
            i_can = abs(cycle_inx[candidate]) - 1
            if cycle_inx[candidate] * cycle_inx[c1] >= 0:  # w tym samym cyklu
                if abs(i_can - i) >= 2 and abs(i_can % (len(cycle1) - 1) - i % (len(cycle1) - 1)) >= 2:
                    con = matrix[c1, candidate]
                    for e, (i1, i2) in enumerate(
                            ((i - 1, i_can - 1), ((i + 1) % len(cycle1), (i_can + 1) % len(cycle1)))):
                        d = con + matrix[cycle1[i1], cycle1[i2]] - matrix[c1, cycle1[i1]] - matrix[
                            candidate, cycle1[i2]]
                        if d < best_d:
                            same_cycle = True
                            best_d = d
                            if e == 0:
                                best_i, best_j = i1, i2
                            else:
                                best_i, best_j = i, i_can

            else:  # w innym cylku
                for i1, i2 in ((i - 1, i - 2), ((i + 1) % len(cycle1), (i + 2) % len(cycle1))):
                    old = matrix[cycle1[i2], cycle1[i1]] + matrix[c1, cycle1[i1]] + \
                          matrix[candidate, cycle2[i_can - 1]] + matrix[candidate, cycle2[(i_can + 1) % len(cycle2)]]
                    new = matrix[cycle1[i2], candidate] + matrix[c1, candidate] + matrix[
                        cycle1[i1], cycle2[i_can - 1]] + matrix[cycle1[i1], cycle2[(i_can + 1) % len(cycle2)]]
                    d = new - old
                    if d < best_d:
                        same_cycle = False
                        best_d = d
                        best_i, best_j = i1, i_can
                        if best_i < 0:
                            best_i = len(cycle1) - 2 - best_i
                        if best_j < 0:
                            best_j = len(cycle2) - 2 - best_j

    return same_cycle, best_i, best_j


if __name__ == "__main__":
    ka200_instance = load_instance('../data/kroa200.tsp')
    kb200_instance = load_instance('../data/krob200.tsp')

    ka200_dm = calc_distance_matrix(ka200_instance)
    kb200_dm = calc_distance_matrix(kb200_instance)

    ka200_can = calculate_candidates(ka200_dm)
    kb200_can = calculate_candidates(kb200_dm)

    ka_cycle1, ka_cycle2 = get_random_cycle(len(ka200_dm))
    kb_cycle1, kb_cycle2 = get_random_cycle(len(kb200_dm))

    print(get_cycles_distance(ka200_dm, ka_cycle1, ka_cycle2))
    print(get_cycles_distance(kb200_dm, kb_cycle1, kb_cycle2))

    ka_cycle1, ka_cycle2 = steep_candidates(ka200_dm, ka_cycle1, ka_cycle2, ka200_can,
                                            is_in_cycle(len(ka200_dm), ka_cycle1, ka_cycle2))
    kb_cycle1, kb_cycle2 = steep_candidates(kb200_dm, kb_cycle1, kb_cycle2, kb200_can,
                                            is_in_cycle(len(kb200_dm), kb_cycle1, kb_cycle2))

    print(get_cycles_distance(ka200_dm, ka_cycle1, ka_cycle2))
    print(get_cycles_distance(kb200_dm, kb_cycle1, kb_cycle2))

    plot_result(ka200_instance, ka_cycle1, ka_cycle2, 'ka200')
    plot_result(kb200_instance, kb_cycle1, kb_cycle2, 'kb200')
