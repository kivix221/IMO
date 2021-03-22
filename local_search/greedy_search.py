from typing import List, Optional
import numpy as np
from IMO.utils import get_random_cycle, load_instance, get_cycles_distance, calc_distance_matrix,plot_result

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

#1)z.w + 2)z.k
def greedy_swap_edges_in_cycle(matrix,cycle1,cycle2):
    search = True
    search_between_cycles = True
    search_in_cycle = True
    while search:
        # Randomizuje typ ruchu czyli albo pomiędzy albo w wewnątrz cyklu
        if choice([True, False]):
            f, s = search_swap_vertices_between_cycle(matrix, cycle1, cycle2)
            if f is None or s is None:
                search_between_cycles = False
            else:
                search_between_cycles = True
                cycle1[f], cycle2[s] = cycle2[s], cycle1[f]
        else:
            #randomizacja cykli -> ruch wewnątrz pierwszego lub drugiego
            if choice([True,False]):
                i,j = search_swap_edges_in_cycle_greedy(matrix,cycle1)
                if i is None or j is None:
                    search_in_cycle = False
                else:
                    search_in_cycle = True
                    cycle1[i+1], cycle1[j] = cycle1[j], cycle1[i+1]
                    cycle1[i + 2:j] = cycle1[i + 2:j][::-1]
            else:
                i,j = search_swap_edges_in_cycle_greedy(matrix,cycle2)
                if i is None or j is None:
                    search_in_cycle = False
                else:
                    search_in_cycle = True
                    cycle2[i+1], cycle2[j] = cycle2[j], cycle2[i+1]
                    cycle2[i + 2:j] = cycle2[i + 2:j][::-1]
        
        search = search_between_cycles or search_in_cycle

    return cycle1,cycle2

def search_swap_edges_in_cycle_greedy(matrix,cycle):
    for i in range(len(cycle) - 1):
        for j in range(i +2, len(cycle) - 1):
            old_distance = matrix[cycle[i]][cycle[i+1]] + matrix[cycle[j]][cycle[j+1]]
            new_distance = matrix[cycle[i]][cycle[j]] + matrix[cycle[i+1]][cycle[j+1]]

            if new_distance - old_distance < 0:
               return i,j
    return None, None


if __name__ == "__main__":
    kroa100_instance = load_instance('data/kroa100.tsp')
    kroa100_distance_matrix = calc_distance_matrix(kroa100_instance)

    k_cycle1, k_cycle2 = get_random_cycle()
    print("Pierwotnie: ", get_cycles_distance(kroa100_distance_matrix, k_cycle1, k_cycle2))
    plot_result(kroa100_instance,k_cycle1,k_cycle2)
    
    v_cycle1, v_cycle2 = k_cycle1.copy(), k_cycle2.copy() #do greedy swap vert in cycle
    v2_cycle1, v2_cycle2 = k_cycle1.copy(), k_cycle2.copy() #do greedy swap vert beetween cycle
    v3_cycle1, v3_cycle2 = k_cycle1.copy(), k_cycle2.copy() #do greedy swap edges

    #swap vertices
    print(len(v_cycle1), len(v_cycle2))
    duration = time.time()
    v_cycle1, v_cycle2 = greedy_swap_vertices_in_cycle(kroa100_distance_matrix, v_cycle1[:-1], v_cycle2[:-1])
    v_cycle1 = np.concatenate([v_cycle1, [v_cycle1[0]]])
    v_cycle2 = np.concatenate([v_cycle2, [v_cycle2[0]]])
    duration = time.time() - duration
    print(len(v_cycle1), len(v_cycle2))
    print("Greedy swap vert in cycle: ",get_cycles_distance(kroa100_distance_matrix, v_cycle1, v_cycle2))
    print("Greedy swap vert in cycle: ",duration)
    plot_result(kroa100_instance,v_cycle1,v_cycle2)
  
    duration = time.time()
    v2_cycle1, v2_cycle2 = greedy_swap_vertices_between_cycle(kroa100_distance_matrix, v2_cycle1[:-1], v2_cycle2[:-1])
    v2_cycle1 = np.concatenate([v2_cycle1, [v2_cycle1[0]]])
    v2_cycle2 = np.concatenate([v2_cycle2, [v2_cycle2[0]]])
    duration = time.time() - duration
    print("Greedy swap vert between cycles: ",get_cycles_distance(kroa100_distance_matrix, v2_cycle1, v2_cycle2))
    print("Greedy swap vert between cycles: ",duration)
    plot_result(kroa100_instance,v2_cycle1,v2_cycle2)
  
    #swap edges
    duration = time.time()
    v3_cycle1, v3_cycle2 = greedy_swap_edges_in_cycle(kroa100_distance_matrix,v3_cycle1,v3_cycle2)
    duration = time.time() - duration
    print("Greedy swap edges: ",get_cycles_distance(kroa100_distance_matrix, v3_cycle1, v3_cycle2))
    print("Greedy swap edges: ",duration)
    plot_result(kroa100_instance,v3_cycle1,v3_cycle2)
