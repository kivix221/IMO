from algo import Algorithm
from utils import *
from time import time
from local_search_2.candidates import calculate_candidates, steep_candidates, is_in_cycle


class Perturbation:
    def perturb(self, cycle1, cycle2, matrix, **kwargs) -> (Iterable, Iterable):
        pass


class SmallPerturbation(Perturbation):
    def perturb(self, cycle1, cycle2, matrix, **kwargs):
        pass


class LargePerturbation(Perturbation):
    def perturb(self, cycle1, cycle2, matrix, **kwargs):
        self.destroy(cycle1, matrix)
        self.destroy(cycle2, matrix)
        self.repair(cycle1, matrix)
        self.repair(cycle2, matrix)

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
