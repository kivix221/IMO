from typing import List, Optional

import numpy as np
from IMO.utils import get_random_cycle, load_instance, get_cycles_distance, calc_distance_matrix
import time
from random import randrange, choice


def search_swap_vertices_in_cycle(matrix, cycle) -> (Optional[int], Optional[int]):
    size = len(cycle)
    it = np.arange(size)
    np.random.shuffle(it)
    for first in it:
        for second in reversed(it):
            b = matrix[cycle[(first - 1)]][cycle[first]] + matrix[cycle[first]][cycle[(first + 1) % size]] \
                + matrix[cycle[(second - 1)]][cycle[second]] + matrix[cycle[second]][cycle[(second + 1) % size]]
            a = matrix[cycle[(first - 1)]][cycle[second]] + matrix[cycle[second]][cycle[(first + 1) % size]] \
                + matrix[cycle[(second - 1)]][cycle[first]] + matrix[cycle[first]][cycle[(second + 1) % size]]
            if a < b:
                return first, second
    return None, None


def greedy_swap_vertices_in_cycle(matrix: np.ndarray, cycle1: np.ndarray, cycle2: np.ndarray) -> (
        List[int], List[int]):
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
    return cycle1, cycle2


def search_swap_vertices_between_cycle(matrix, cycle1, cycle2) -> (Optional[int], Optional[int]):
    size1, size2 = len(cycle1), len(cycle2)
    it1, it2 = np.arange(size1), np.arange(size2)
    np.random.shuffle(it1)
    np.random.shuffle(it2)
    if choice([True, False]):
        for first in it1:
            for second in it2:
                b = matrix[cycle1[(first - 1)]][cycle1[first]] + matrix[cycle1[first]][cycle1[(first + 1) % size1]] \
                    + matrix[cycle2[second - 1]][cycle2[second]] + matrix[cycle2[second]][cycle2[(second + 1) % size2]]
                a = matrix[cycle1[(first - 1)]][cycle2[second]] + matrix[cycle2[second]][cycle1[(first + 1) % size1]] \
                    + matrix[cycle2[(second - 1)]][cycle1[first]] + matrix[cycle1[first]][cycle2[(second + 1) % size2]]
                if a < b:
                    return first, second
    else:
        for second in it2:
            for first in it1:
                b = matrix[cycle1[(first - 1)]][cycle1[first]] + matrix[cycle1[first]][cycle1[(first + 1) % size1]] \
                    + matrix[cycle2[second - 1]][cycle2[second]] + matrix[cycle2[second]][cycle2[(second + 1) % size2]]
                a = matrix[cycle1[(first - 1)]][cycle2[second]] + matrix[cycle2[second]][cycle1[(first + 1) % size1]] \
                    + matrix[cycle2[(second - 1)]][cycle1[first]] + matrix[cycle1[first]][cycle2[(second + 1) % size2]]
                if a < b:
                    return first, second
    return None, None


def greedy_swap_vertices_between_cycle(matrix: np.ndarray, cycle1: np.ndarray, cycle2: np.ndarray) -> (
        List[int], List[int]):
    search = True
    while search:
        search = False
        f, s = search_swap_vertices_between_cycle(matrix, cycle1, cycle2)
        if f is not None:
            search = True
            cycle1[f], cycle2[s] = cycle2[s], cycle1[f]
    return cycle1, cycle2


def greedy_swap_edges_in_cycle(matrix: np.ndarray, cycle1: np.ndarray, cycle2: np.ndarray) -> (
        List[int], List[int]):
    pass


def greedy_swap_edges_between_cycle(matrix: np.ndarray, cycle1: np.ndarray, cycle2: np.ndarray) -> (
        List[int], List[int]):
    pass


if __name__ == "__main__":
    kroa100_instance = load_instance('data/kroa100.tsp')
    kroa100_distance_matrix = calc_distance_matrix(kroa100_instance)

    k_cycle1, k_cycle2 = get_random_cycle()
    v_cycle1, v_cycle2 = k_cycle1.copy(), k_cycle2.copy()
    print(get_cycles_distance(kroa100_distance_matrix, k_cycle1, k_cycle2))

    v_cycle1, v_cycle2 = greedy_swap_vertices_in_cycle(kroa100_distance_matrix, v_cycle1, v_cycle2)
    duration = time.time()
    k_cycle1, k_cycle2 = greedy_swap_vertices_between_cycle(kroa100_distance_matrix, k_cycle1, k_cycle2)
    duration = time.time() - duration
    print(get_cycles_distance(kroa100_distance_matrix, v_cycle1, v_cycle2))
    print(get_cycles_distance(kroa100_distance_matrix, k_cycle1, k_cycle2))
    print(duration)
