from algo import Algorithm
from utils import *
from time import time
from local_search_2.candidates import calculate_candidates, steep_candidates, is_in_cycle
from random import randrange
import numpy as np


class Perturbation:
    def perturb(self, cycle1, cycle2, matrix, **kwargs) -> (Iterable, Iterable):
        pass


class SmallPerturbation(Perturbation):
    def __init__(self, n):
        self.n = n

    def _swap_vert_inside(self, cycle1, cycle2):
        if np.random.choice([True, False], 1)[0]:
            c = cycle1
        else:
            c = cycle2
        s1, s2 = np.random.choice(len(c) - 1, 2, False)
        c[s1], c[s2] = c[s2], c[s1]
        c[-1] = c[0]
        return cycle1, cycle2

    def _swap_vert_between(self, cycle1, cycle2):
        s1, s2 = np.random.choice(len(cycle1) - 1, 2)
        cycle1[s1], cycle2[s2] = cycle2[s2], cycle1[s1]
        cycle1[-1], cycle2[-1] = cycle1[0], cycle2[0]
        return cycle1, cycle2

    def _swap_edge_inside(self, cycle1, cycle2):
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

    _swap_func = (_swap_vert_between, _swap_vert_inside, _swap_edge_inside)

    def perturb(self, cycle1, cycle2, matrix, **kwargs) -> (Iterable, Iterable):
        c1, c2 = np.copy(cycle1), np.copy(cycle2)
        for _ in range(self.n):
            c1, c2 = self._swap_func[randrange(len(self._swap_func))](c1, c2)
        return c1, c2


class LargePerturbation(Perturbation):
    def perturb(self, cycle1, cycle2, matrix, **kwargs) -> (Iterable, Iterable):
        self.destroy(cycle1, matrix)
        self.destroy(cycle2, matrix)
        self.repair(cycle1, matrix)
        self.repair(cycle2, matrix)
        return cycle1, cycle2

    def destroy(self, cycle, matrix):
        pass

    def repair(self, cycle, matrix):
        pass


class IteratedLS(Algorithm):
    def __init__(self, matrix):
        super().__init__(matrix)
        self.candidates = calculate_candidates(matrix)

    def _run_steep_can(self, cycle1, cycle2):
        return steep_candidates(self.matrix, cycle1, cycle2, self.candidates,
                                is_in_cycle(self.n, cycle1, cycle2))

    def run(self, p: Perturbation, stop_time=1.5, **kwargs):
        t, t_dur = time(), 0.0

        cycle1, cycle2 = get_random_cycle(self.n)

        best_c1, best_c2 = cycle1, cycle2
        best_d = get_cycles_distance(self.matrix, best_c1, best_c2)

        while t_dur < stop_time:
            nc1, nc2 = p.perturb(cycle1, cycle2, self.matrix)

            nc1, nc2 = self._run_steep_can(nc1, nc2)

            new_d = get_cycles_distance(self.matrix, nc1, nc2)  # MOŻLIWE ULEPSZENIE W WYLICZANIU DYSTANSU
            if new_d < best_d:
                best_d = new_d
                best_c1, best_c2 = nc1, nc2

        return best_c1, best_c2
