import random
from typing import List

from numba import njit, objmode
from time import time
# from random import randrange
import random
import numpy as np
from utils import plot_result, load_instance, calc_distance_matrix
import ray
import asyncio
from tqdm import tqdm


# try:
#     from .algo import Algorithm
#     from ..utils import *
#     from ..local_search_2.candidates import calculate_candidates, steep_candidates, is_in_cycle
#     from ..greedy_heuristics.regret_cycle import get_next_regret_node, get_cycle_next_node, double_regret_cycle
# except Exception:
# from algo import Algorithm
# from utils import *


@njit
def get_random_cycle(n=100):
    whole = np.arange(n)
    np.random.shuffle(whole)

    # stara wersja
    # return np.array(whole[(n//2+1):]), np.array(whole[:(n//2+1)])

    # moja propozycja ~ Marcin
    cycle1, cycle2 = np.zeros((n // 2 + 1,), dtype=np.int64), np.zeros((n // 2 + 1,), dtype=np.int64)
    cycle1[:-1] = whole[(n // 2):]
    cycle2[:-1] = whole[:(n // 2)]
    cycle1[-1] = cycle1[0]
    cycle2[-1] = cycle2[0]
    return cycle1, cycle2


@njit
def get_cycle_distance(matrix: np.ndarray, cycle: np.ndarray):
    distance = 0.0
    # for pr, nx in zip(cycle[:-1], cycle[1:]):
    #     distance += matrix[pr][nx]
    for i in range(len(cycle) - 1):
        c1 = cycle[i]
        c2 = cycle[i + 1]
        d = matrix[c1][c2]
        distance += d

    return distance


@njit
def get_cycles_distance(matrix: np.ndarray, cycle1: np.ndarray, cycle2: np.ndarray):
    dis1 = get_cycle_distance(matrix, cycle1)
    dis2 = get_cycle_distance(matrix, cycle2)
    return dis1 + dis2, dis1, dis2


@njit
def get_first_node(matrix: np.ndarray, not_been: list, node=0) -> int:
    """
    zwraca najbliższy nieodwiedzony wierzchołek
    """
    i, m = -1, np.inf
    for v in not_been:
        if matrix[v, node] < m:
            m = matrix[v, node]
            i = v
    return i


@njit
def get_cycle_next_node(matrix: np.ndarray, not_been: list, cycle: list) -> (int, int):
    """
    zwraca niewykorzystany jeszcze wierzchołek, którego dodanie do podanego cyklu
    spowoduje najmniejsze zwiększenie dystansu
    """
    mi, mv = 0, np.inf
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


@njit
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


@njit
def list2np(l):
    new = np.zeros((len(l),), dtype=np.int64)
    for i, v in enumerate(l):
        new[i] = v
    return new


@njit
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
    # return np.full((len(cycle1),), cycle1, np.int64), np.full((len(cycle2),), cycle2, np.int64)
    # return np.array(cycle1).astype(np.int64), np.array(cycle2).astype(np.int64)
    return list2np(cycle1), list2np(cycle2)


@njit
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


@njit
def is_in_cycle(n: int, cycle1: np.ndarray, cycle2: np.ndarray):
    in_cycle = np.zeros((n,), dtype=np.int64)
    for i, c in enumerate(cycle1[:-1]):
        in_cycle[c] = i + 1
    for i, c in enumerate(cycle2[:-1]):
        in_cycle[c] = - i - 1
    return in_cycle


# @njit
# def is_in_cycle(n: int, cycle1: np.ndarray, cycle2: np.ndarray):
#     in_cycle = np.zeros((n,), dtype=np.int32)
#     for i in range(len(cycle1) - 1):
#         in_cycle[cycle1[i]] = i + 1
#     for i in range(len(cycle2) - 1):
#         in_cycle[cycle2[i]] = -i - 1
#     return in_cycle

@njit
def update_in_cycle(in_cycle: np.ndarray, cycle: np.ndarray, mul: int):
    for i, v in enumerate(cycle[:-1]):
        in_cycle[v] = mul * (i + 1)
    return in_cycle


@njit
def steep_candidates(matrix, cycle1, cycle2, candidates, cycle_inx):
    delt = 0
    search = True
    while search:
        search = False
        s, i, j, d = search_swap_in_candidates(matrix, cycle1[:-1], cycle2[:-1], candidates, cycle_inx)
        if i != -1 and j != -1:
            delt += d
            il, jl = i, j
            search = True
            if s:
                il, jl = min(il, jl), max(il, jl)
                cycle1[il + 1:jl + 1] = cycle1[il + 1:jl + 1][::-1]
                cycle_inx = update_in_cycle(cycle_inx, cycle1, 1)
            else:
                tmp = cycle1[il]
                cycle1[il] = cycle2[jl]
                cycle2[jl] = tmp
                cycle_inx[cycle1[il]] = i + 1
                cycle_inx[cycle2[jl]] = -j - 1

    search = True
    while search:
        search = False
        s, i, j, d = search_swap_in_candidates(matrix, cycle2[:-1], cycle1[:-1], candidates, cycle_inx)
        if i != -1 or j != -1:
            delt += d
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
    return cycle1, cycle2, delt


@njit
def search_swap_in_candidates(matrix, cycle1, cycle2, candidates, cycle_inx):
    best_i, best_j = -1, -1
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

    return same_cycle, best_i, best_j, best_d


class Perturbation:
    @njit
    def perturb(self, cycle1, cycle2, matrix, size, **kwargs):
        return np.copy(cycle1), np.copy(cycle2)


@njit
def _swap_vert_inside(cycle1, cycle2, matrix):
    if random.randrange(2) == 0:
        c = cycle1
    else:
        c = cycle2
    s1, s2 = np.random.choice(len(c) - 1, 2, False)
    bef = matrix[c[s1], c[s1 - 1]] + matrix[c[s1], c[(s1 + 1)]] + \
          matrix[c[s2], c[s2 - 1]] + matrix[c[s2], c[(s2 + 1)]]
    c[s1], c[s2] = c[s2], c[s1]
    c[-1] = c[0]
    aft = matrix[c[s1], c[s1 - 1]] + matrix[c[s1], c[(s1 + 1) % len(c)]] + \
          matrix[c[s2], c[s2 - 1]] + matrix[c[s2], c[(s2 + 1) % len(c)]]
    return cycle1, cycle2, aft - bef


@njit
def _swap_vert_between(cycle1, cycle2, matrix):
    s1, s2 = np.random.choice(len(cycle1) - 1, 2)
    bef = matrix[cycle1[s1], cycle1[s1 - 1]] + matrix[cycle1[s1], cycle1[(s1 + 1)]] + \
          matrix[cycle2[s2], cycle2[s2 - 1]] + matrix[cycle2[s2], cycle2[(s2 + 1)]]
    cycle1[s1], cycle2[s2] = cycle2[s2], cycle1[s1]
    cycle1[-1], cycle2[-1] = cycle1[0], cycle2[0]
    aft = matrix[cycle1[s1], cycle1[s1 - 1]] + matrix[cycle1[s1], cycle1[(s1 + 1)]] + \
          matrix[cycle2[s2], cycle2[s2 - 1]] + matrix[cycle2[s2], cycle2[(s2 + 1)]]
    return cycle1, cycle2, aft - bef


@njit
def _swap_edge_inside(cycle1, cycle2, matrix):
    if random.randrange(2) == 0:
        c = cycle1
    else:
        c = cycle2
    s1, s2 = np.random.choice(len(c) - 1, 2, False)
    while abs(s1 - s2) < 3:
        s1, s2 = np.random.choice(len(c) - 1, 2, False)
    s1, s2 = min(s1, s2), max(s1, s2)
    bef = matrix[c[s1 - 1], c[s1]] + matrix[c[s2], c[s2 + 1]]
    c[s1:s2 + 1] = c[s1:s2 + 1][::-1]
    c[-1] = c[0]
    aft = matrix[c[s1 - 1], c[s1]] + matrix[c[s2], c[s2 + 1]]
    return cycle1, cycle2, aft - bef


@njit
def perturb(cycle1, cycle2, size, matrix):
    c1, c2 = np.copy(cycle1), np.copy(cycle2)
    distance = 0
    for _ in range(size):
        r = random.randrange(3)
        if r == 0:
            c1, c2, d = _swap_vert_between(c1, c2, matrix)
        elif r == 1:
            c1, c2, d = _swap_vert_inside(c1, c2, matrix)
        else:
            c1, c2, d = _swap_edge_inside(c1, c2, matrix)
        distance += d
    return c1, c2, distance


# class LargePerturbation(Perturbation):
#     @njit
#     def perturb(self, cycle1, cycle2, matrix, size, instance=None, rand=0.0, **kwargs) -> (Iterable, Iterable):
#         assert instance is not None
#         assert 0.0 <= size <= 1.0
#         assert 0.0 <= rand <= 1.0
#         c1, c2 = super(LargePerturbation, self).perturb(cycle1, cycle2, matrix, size)
#
#         middles = self._calculate_middles(c1, c2, instance)
#
#         c1, c2, removed = self.destroy(c1, c2, size, rand, instance, middles)
#         c1, c2 = self.repair(c1, c2, matrix, removed)
#         return c1, c2
#
#     @staticmethod
#     @njit
#     def _calculate_middles(cycle1, cycle2, instance):
#         m1 = np.mean(instance[cycle1[:-1] - 1], axis=0)
#         m2 = np.mean(instance[cycle2[:-1] - 1], axis=0)
#         return m1, m2
#
#     @staticmethod
#     @njit
#     def destroy(cycle1, cycle2, size, rand, instance, middles):
#         removed = np.array([], dtype=np.int32)
#
#         d1 = np.sum((instance[cycle1[:-1] - 1] - middles[0]) ** 2, axis=1)
#         d2 = np.sum((instance[cycle2[:-1] - 1] - middles[1]) ** 2, axis=1)
#         d1 = np.stack([d1, np.arange(d1.shape[0])], axis=-1)
#         d2 = np.stack([d2, np.arange(d2.shape[0])], axis=-1)
#         d1 = np.array(sorted(d1, key=lambda x: x[0], reverse=True), dtype=int)
#         d2 = np.array(sorted(d2, key=lambda x: x[0], reverse=True), dtype=int)
#
#         qn = int(size * (1. - rand) * instance.shape[0])  # wyrzucanie wierzchołków najdalszych od środków
#         removed = np.concatenate([removed, cycle1[d1[:qn // 2, 1]], cycle2[d2[:qn // 2, 1]]], axis=0)
#         cycle1 = np.delete(cycle1, d1[:qn // 2, 1])
#         cycle2 = np.delete(cycle2, d2[:qn // 2, 1])
#
#         qn = int(size * rand * instance.shape[0])  # wyrzucanie losowych wierzchołków
#         to_rem1 = np.random.choice(cycle1.shape[0] - 1, qn // 2, False)
#         to_rem2 = np.random.choice(cycle2.shape[0] - 1, qn // 2, False)
#         removed = np.concatenate([removed, cycle1[to_rem1], cycle2[to_rem2]], axis=0)
#         cycle1 = np.delete(cycle1, to_rem1)
#         cycle2 = np.delete(cycle2, to_rem2)
#
#         return cycle1, cycle2, removed
#
#     @staticmethod
#     @njit
#     def repair(cycle1, cycle2, matrix, removed):
#         s2 = matrix.shape[0] // 2
#         s1 = s2 + matrix.shape[0] % 2
#
#         c1, c2 = cycle1[:-1], cycle2[:-1]
#
#         while c1.shape[0] < s1 and c2.shape[0] < s2:
#             removed_ind, cycle_ind, cyc = get_next_regret_node(matrix, removed, c1, c2)
#             if not cyc:
#                 c1 = np.insert(c1, cycle_ind + 1, removed[removed_ind])
#             else:
#                 c2 = np.insert(c2, cycle_ind + 1, removed[removed_ind])
#             removed = np.delete(removed, removed_ind)
#
#         while c1.shape[0] < s1:
#             r_ind, cyc_ind = get_cycle_next_node(matrix, removed, c1)
#             c1 = np.insert(c1, cyc_ind + 1, removed[r_ind])
#             removed = np.delete(removed, r_ind)
#
#         while c2.shape[0] < s2:
#             r_ind, cyc_ind = get_cycle_next_node(matrix, removed, c2)
#             c2 = np.insert(c2, cyc_ind + 1, removed[r_ind])
#             removed = np.delete(removed, r_ind)
#
#         c1, c2 = np.append(c1, [c1[0]]), np.append(c2, [c2[0]])
#         return c1, c2


@ray.remote
class ASet:
    def __init__(self):
        self.been = set()

    async def check_score(self, t: int):
        if t in self.been:
            return True
        else:
            self.been.add(t)
            return False

    async def add_score(self, t: int):
        self.been.add(t)


@njit
def _regret_begin(matrix, n):
    return double_regret_cycle(matrix, random.randrange(n))


@njit
def _ls_begin(matrix, candidates, n):
    cycle1, cycle2 = get_random_cycle(n)
    return steep_candidates(matrix, cycle1, cycle2, candidates,
                            is_in_cycle(n, cycle1, cycle2))


# @njit
@ray.remote
def run_algo(size, matrix, candidates, stop_time=1.5, regret_begin=False, n=200):
    iteration = 0
    found = 0
    # with objmode(t='f8'):
    t = time()
    t_dur = 0.0

    if regret_begin:
        c1, c2 = _regret_begin(matrix, n)
    else:
        c1, c2, _ = _ls_begin(matrix, candidates, n)
    best_c1, best_c2 = c1, c2
    best_d = get_cycles_distance(matrix, best_c1, best_c2)[0]

    while t_dur < stop_time:
        iteration += 1
        nc1, nc2, delt = perturb(best_c1, best_c2, size, matrix)

        nc1, nc2, d = steep_candidates(matrix, nc1, nc2, candidates,
                                       is_in_cycle(n, nc1, nc2))

        if delt + d < 0:
            found += 1
            best_d += delt + d
            best_c1, best_c2 = nc1, nc2

        # with objmode(tk='f8'):
        tk = time()
        t_dur = tk - t

    return (best_c1, best_c2), (found / iteration, iteration), best_d


@ray.remote
def _run_parallel(size, matrix, candidates, stop_time=1.5, regret_begin=False, n=200):
    return run_algo(size, matrix, candidates, stop_time, regret_begin, n)


@njit
def calc_from_results(results, matrix: np.ndarray):
    # cyc = [_[0] for _ in results]
    # stat = [_[1] for _ in results]
    s = np.inf
    si = -1
    for i, c in enumerate(results):
        if c[2] < s:
            s = c[2]
            si = i
    z, w = 0, 0
    for _, st, _ in results:
        z += st[0] * st[1]
        w += st[1]
    return results[si][0], (float(z) / w, w), results[si][2]


def run_parallel(size, matrix, candidates, stop_time=1.5, regret_begin=False, n=200):
    results = [run_algo.remote(size, matrix, candidates, stop_time, regret_begin, n) for _ in range(4)]
    results = ray.get(results)
    return calc_from_results(results, matrix)


if __name__ == "__main__":
    ka200_instance = load_instance('../data/kroa200.tsp')
    kb200_instance = load_instance('../data/krob200.tsp')

    ka200_dm = calc_distance_matrix(ka200_instance)
    kb200_dm = calc_distance_matrix(kb200_instance)

    candidates_ka = calculate_candidates(ka200_dm)
    candidates_kb = calculate_candidates(kb200_dm)

    # run_algo(7, kb200_dm, candidates_ka, 10)

    ray.init()
    data = []
    cycy = []
    aset = ASet.remote()
    for _ in tqdm(range(10)):
        (cy1, cy2), _, best = run_parallel(7, ka200_dm, candidates_ka, stop_time=150,
                                           regret_begin=True)  # , rand=1.0, instance=kb200_instance)
        # dist = get_cycles_distance(kb200_dm, cy1, cy2)
        # if dist[0] != best:
        #     print("!!!!!!!!!!!!!!!!!", dist[0], best)
        data.append((best, *_))
        cycy.append((cy1, cy2))
    data = np.array(data)
    print("maxy:")
    print(np.max(data, axis=0))
    print("meany:")
    print(np.mean(data, axis=0))
    print("miny:")
    print(np.min(data, axis=0))

    b = cycy[np.argmin(data[:, 0])]

    # print(f"Iteracji: {_[1]}\nProcent znalezionych lepszych: {_[0]*100}%")
    # b = [0,0]
    # b[0] = np.fromfile('../../c1_kb_29714.csv', dtype=np.int64, sep='\n')
    # b[1] = np.fromfile('../../c2_kb_29714.csv', dtype=np.int64, sep='\n')
    # print(b[0].shape)
    dist = get_cycles_distance(ka200_dm, b[0], b[1])
    # print(dist)
    # plot_result(kb200_instance, b[0], b[1], str(dist[0]))
    plot_result(ka200_instance, b[0], b[1], str(dist[0]))
