from typing import List, Optional

import numpy as np
from utils import get_random_cycle, load_instance, get_cycles_distance, calc_distance_matrix
import time
from random import randrange


def search_swap_vertices_in_cycle(matrix, cycle) -> (Optional[int], Optional[int]):
    size = len(cycle)
    it = np.arange(size)
    np.random.shuffle(it)
    for first in it:
        for second in reversed(it):
            b = matrix[cycle[(first - 1) % size]][cycle[first]] + matrix[cycle[first]][cycle[(first + 1) % size]] \
                + matrix[cycle[(second - 1) % size]][cycle[second]] + matrix[cycle[second]][cycle[(second + 1) % size]]
            a = matrix[cycle[(first - 1) % size]][cycle[second]] + matrix[cycle[second]][cycle[(first + 1) % size]] \
                + matrix[cycle[(second - 1) % size]][cycle[first]] + matrix[cycle[first]][cycle[(second + 1) % size]]
            if a < b:
                return first, second
    return None, None


def greedy_swap_vertices_in_cycle(matrix, cycle1, cycle2) -> (List[int], List[int]):
    search = True
    while search:
        search = False
        f, s = search_swap_vertices_in_cycle(matrix, cycle1)
        if f is not None:
            search = True
            cycle1[f], cycle1[s] = cycle1[s], cycle1[f]
    search = True
    while search:
        search = False
        f, s = search_swap_vertices_in_cycle(matrix, cycle2)
        if f is not None:
            search = True
            cycle2[f], cycle2[s] = cycle2[s], cycle2[f]


def greedy_swap_vertices_between_cycle(matrix, cycle1, cycle2):
    pass


def greedy_swap_edges_in_cycle(matrix, cycle1, cycle2):
    pass


def greedy_swap_edges_between_cycle(matrix, cycle1, cycle2):
    pass


if __name__ == "__main__":
    kroa100_instance = load_instance('kroa100.tsp')
    kroa100_distance_matrix = calc_distance_matrix(kroa100_instance)

    k_cycle1, k_cycle2 = get_random_cycle()
    print(get_cycles_distance(kroa100_distance_matrix, k_cycle1, k_cycle2))

    duration = time.time()
    greedy_swap_vertices_in_cycle(kroa100_distance_matrix, k_cycle1, k_cycle2)
    duration = time.time() - duration
    print(get_cycles_distance(kroa100_distance_matrix, k_cycle1, k_cycle2))
    print(duration)
