import random
from time import time
from random import randrange
try:
    from .algo import Algorithm
    from ..utils import *
    from ..local_search_2.candidates import calculate_candidates, steep_candidates, is_in_cycle
    from ..greedy_heuristics.regret_cycle import get_next_regret_node, get_cycle_next_node, double_regret_cycle
except Exception:
    from algo import Algorithm
    from utils import *
    from local_search_2.candidates import calculate_candidates, steep_candidates, is_in_cycle
    from greedy_heuristics.regret_cycle import get_next_regret_node, get_cycle_next_node, double_regret_cycle


class Perturbation:
    def perturb(self, cycle1, cycle2, matrix, size, **kwargs) -> (Iterable, Iterable):
        return np.copy(cycle1), np.copy(cycle2)


class SmallPerturbation(Perturbation):
    @staticmethod
    def _swap_vert_inside(cycle1, cycle2):
        if np.random.choice([True, False], 1)[0]:
            c = cycle1
        else:
            c = cycle2
        s1, s2 = np.random.choice(len(c) - 1, 2, False)
        c[s1], c[s2] = c[s2], c[s1]
        c[-1] = c[0]
        return cycle1, cycle2

    @staticmethod
    def _swap_vert_between(cycle1, cycle2):
        s1, s2 = np.random.choice(len(cycle1) - 1, 2)
        cycle1[s1], cycle2[s2] = cycle2[s2], cycle1[s1]
        cycle1[-1], cycle2[-1] = cycle1[0], cycle2[0]
        return cycle1, cycle2

    @staticmethod
    def _swap_edge_inside(cycle1, cycle2):
        if np.random.choice([True, False], 1)[0]:
            c = cycle1
        else:
            c = cycle2
        s1, s2 = np.random.choice(len(c) - 1, 2, False)
        while abs(s1 - s2) < 3:
            s1, s2 = np.random.choice(len(c) - 1, 2, False)
        s1, s2 = min(s1, s2), max(s1, s2)
        c[s1:s2 + 1] = c[s1:s2 + 1][::-1]
        c[-1] = c[0]
        return cycle1, cycle2

    _swap_func = (_swap_vert_between.__func__, _swap_vert_inside.__func__,
                  _swap_edge_inside.__func__)

    def perturb(self, cycle1, cycle2, matrix, size, **kwargs) -> (Iterable, Iterable):
        c1, c2 = super(SmallPerturbation, self).perturb(cycle1, cycle2, matrix, size)
        for _ in range(size):
            c1, c2 = self._swap_func[randrange(len(self._swap_func))](c1, c2)
        return c1, c2


class LargePerturbation(Perturbation):
    def perturb(self, cycle1, cycle2, matrix, size, instance=None, rand=0.0, **kwargs) -> (Iterable, Iterable):
        assert instance is not None
        assert 0.0 <= size <= 1.0
        assert 0.0 <= rand <= 1.0
        c1, c2 = super(LargePerturbation, self).perturb(cycle1, cycle2, matrix, size)

        middles = self._calculate_middles(c1, c2, instance)

        c1, c2, removed = self.destroy(c1, c2, size, rand, instance, middles)
        c1, c2 = self.repair(c1, c2, matrix, removed)
        return c1, c2

    @staticmethod
    def _calculate_middles(cycle1, cycle2, instance):
        m1 = np.mean(instance[cycle1[:-1] - 1], axis=0)
        m2 = np.mean(instance[cycle2[:-1] - 1], axis=0)
        return m1, m2

    @staticmethod
    def destroy(cycle1, cycle2, size, rand, instance, middles):
        removed = np.array([], dtype=np.int32)

        d1 = np.sum((instance[cycle1[:-1] - 1] - middles[0]) ** 2, axis=1)
        d2 = np.sum((instance[cycle2[:-1] - 1] - middles[1]) ** 2, axis=1)
        d1 = np.stack([d1, np.arange(d1.shape[0])], axis=-1)
        d2 = np.stack([d2, np.arange(d2.shape[0])], axis=-1)
        d1 = np.array(sorted(d1, key=lambda x: x[0], reverse=True), dtype=int)
        d2 = np.array(sorted(d2, key=lambda x: x[0], reverse=True), dtype=int)

        qn = int(size * (1. - rand) * instance.shape[0])  # wyrzucanie wierzchołków najdalszych od środków
        removed = np.concatenate([removed, cycle1[d1[:qn // 2, 1]], cycle2[d2[:qn // 2, 1]]], axis=0)
        cycle1 = np.delete(cycle1, d1[:qn // 2, 1])
        cycle2 = np.delete(cycle2, d2[:qn // 2, 1])

        qn = int(size * rand * instance.shape[0])  # wyrzucanie losowych wierzchołków
        to_rem1 = np.random.choice(cycle1.shape[0] - 1, qn // 2, False)
        to_rem2 = np.random.choice(cycle2.shape[0] - 1, qn // 2, False)
        removed = np.concatenate([removed, cycle1[to_rem1], cycle2[to_rem2]], axis=0)
        cycle1 = np.delete(cycle1, to_rem1)
        cycle2 = np.delete(cycle2, to_rem2)

        return cycle1, cycle2, removed

    @staticmethod
    def repair(cycle1, cycle2, matrix, removed):
        s2 = matrix.shape[0] // 2
        s1 = s2 + matrix.shape[0] % 2

        c1, c2 = cycle1[:-1], cycle2[:-1]

        while c1.shape[0] < s1 and c2.shape[0] < s2:
            removed_ind, cycle_ind, cyc = get_next_regret_node(matrix, removed, c1, c2)
            if not cyc:
                c1 = np.insert(c1, cycle_ind + 1, removed[removed_ind])
            else:
                c2 = np.insert(c2, cycle_ind + 1, removed[removed_ind])
            removed = np.delete(removed, removed_ind)

        while c1.shape[0] < s1:
            r_ind, cyc_ind = get_cycle_next_node(matrix, removed, c1)
            c1 = np.insert(c1, cyc_ind + 1, removed[r_ind])
            removed = np.delete(removed, r_ind)

        while c2.shape[0] < s2:
            r_ind, cyc_ind = get_cycle_next_node(matrix, removed, c2)
            c2 = np.insert(c2, cyc_ind + 1, removed[r_ind])
            removed = np.delete(removed, r_ind)

        c1, c2 = np.append(c1, [c1[0]]), np.append(c2, [c2[0]])
        return c1, c2


class IteratedLSa(Algorithm):
    def __init__(self, matrix):
        super().__init__(matrix)
        self.candidates = calculate_candidates(matrix)

    def __str__(self):
        return 'ILSa'

    def _run_steep_can(self, cycle1, cycle2):
        return steep_candidates(self.matrix, cycle1, cycle2, self.candidates,
                                is_in_cycle(self.n, cycle1, cycle2))

    def _regret_begin(self):
        return double_regret_cycle(self.matrix, random.randrange(self.n))

    def _ls_begin(self):
        cycle1, cycle2 = get_random_cycle(self.n)
        return self._run_steep_can(cycle1, cycle2)

    def run(self, p: Perturbation, size, stop_time=1.5, regret_begin=False, **kwargs):
        iteration = 0
        found = 0
        t, t_dur = time(), 0.0

        if regret_begin:
            best_c1, best_c2 = self._regret_begin()
        else:
            best_c1, best_c2 = self._ls_begin()
        best_d = get_cycles_distance(self.matrix, best_c1, best_c2)

        while t_dur < stop_time:
            iteration += 1
            nc1, nc2 = p.perturb(best_c1, best_c2, self.matrix, size, **kwargs)

            new_d = get_cycles_distance(self.matrix, nc1, nc2)  # MOŻLIWE ULEPSZENIE W WYLICZANIU DYSTANSU
            if new_d < best_d:
                found += 1
                best_d = new_d
                best_c1, best_c2 = nc1, nc2

            t_dur = time() - t

        return (best_c1, best_c2), (found/iteration, iteration)


class IteratedLS(IteratedLSa):
    def __str__(self):
        return 'ILS'

    def run(self, p: Perturbation, size, stop_time=1.5, regret_begin=False, **kwargs):
        iteration = 0
        found = 0
        t, t_dur = time(), 0.0

        if regret_begin:
            best_c1, best_c2 = self._regret_begin()
        else:
            best_c1, best_c2 = self._ls_begin()
        best_d = get_cycles_distance(self.matrix, best_c1, best_c2)

        while t_dur < stop_time:
            iteration += 1
            nc1, nc2 = p.perturb(best_c1, best_c2, self.matrix, size, **kwargs)

            nc1, nc2 = self._run_steep_can(nc1, nc2)

            new_d = get_cycles_distance(self.matrix, nc1, nc2)  # MOŻLIWE ULEPSZENIE W WYLICZANIU DYSTANSU
            if new_d < best_d:
                found += 1
                best_d = new_d
                best_c1, best_c2 = nc1, nc2

            t_dur = time() - t

        return (best_c1, best_c2), (found/iteration, iteration)


if __name__ == "__main__":
    ka200_instance = load_instance('IMO/data/kroa200.tsp')
    kb200_instance = load_instance('IMO/data/krob200.tsp')

    ka200_dm = calc_distance_matrix(ka200_instance)
    kb200_dm = calc_distance_matrix(kb200_instance)

    it = IteratedLS(kb200_dm)
    s_per = SmallPerturbation()
    l_per = LargePerturbation()
    cy1, cy2 = it.run(l_per, stop_time=15, size=0.2, rand=1.0, instance=kb200_instance)

    dist = get_cycles_distance(kb200_dm, cy1, cy2)
    print(get_cycles_distance(kb200_dm, cy1, cy2))
    plot_result(kb200_instance, cy1, cy2, str(dist[0]))
