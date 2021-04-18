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
    for i, c in enumerate(cycle1):
        in_cycle[c] = i
    for i, c in enumerate(cycle2):
        in_cycle[c] = -i
    return in_cycle


def steep_candidates(matrix, cycle1, cycle2, candidates, cycle_inx):
    search = True
    while search:
        search = False
        s, i, j = search_swap_in_candidates(matrix, cycle1, cycle2, candidates, cycle_inx)
        if i is not None or j is not None:
            search = True
            if s:
                cycle1[i + 1], cycle1[j] = cycle1[j], cycle1[i + 1]
                cycle1[i + 2:j] = cycle1[i + 2:j][::-1]
            else:
                cycle1[i], cycle2[j] = cycle2[j], cycle1[i]
                cycle_inx[cycle1[i]] = -i
                cycle_inx[cycle2[j]] = j

        s, i, j = search_swap_in_candidates(matrix, cycle2, cycle1, candidates, cycle_inx)
        if i is not None or j is not None:
            search = True
            if s:
                cycle2[i + 1], cycle2[j] = cycle2[j], cycle2[i + 1]
                cycle2[i + 2:j] = cycle2[i + 2:j][::-1]
            else:
                cycle2[i], cycle1[j] = cycle1[j], cycle2[i]
                cycle_inx[cycle2[i]] = -i
                cycle_inx[cycle1[j]] = j


    return cycle1, cycle2


def search_swap_in_candidates(matrix, cycle1, cycle2, candidates, cycle_inx):
    best_i, best_j = None, None
    best_d = np.inf
    same_cycle = True
    for i, c1 in enumerate(cycle1):
        for candidate in candidates[c1]:
            if cycle_inx[candidate] * cycle_inx[c1] >= 0:  # w tym samym cyklu
                pass

            else:  # w innym cylku
                for i1, i2 in ((i-1, i-2), ((i+1)%len(cycle1), (i+2)%len(cycle1))):
                    old = matrix[cycle1[i2], cycle1[i1]] + matrix[c1, cycle1[i1]] + \
                          matrix[candidate, cycle2[abs(cycle_inx[candidate]) - 1]] + \
                          matrix[candidate, cycle2[(abs(cycle_inx[candidate]) + 1) % len(cycle2)]]
                    new = matrix[cycle1[i2], candidate] + matrix[c1, candidate] + \
                          matrix[cycle1[i1], cycle2[abs(cycle_inx[candidate]) - 1]] + \
                          matrix[cycle1[i1], cycle2[(abs(cycle_inx[candidate]) + 1) % len(cycle2)]]
                    d = new-old
                    if d < best_d:
                        same_cycle = False
                        best_d = d
                        best_i, best_j = i1, abs(cycle_inx[candidate])

    return same_cycle, best_i, best_j


if __name__ == "__main__":
    ka200_instance = load_instance('../data/kroa200.tsp')
    kb200_instance = load_instance('../data/krob200.tsp')

    ka200_dm = calc_distance_matrix(ka200_instance)
    kb200_dm = calc_distance_matrix(kb200_instance)

    calculate_candidates(ka200_dm)
    calculate_candidates(kb200_dm)
