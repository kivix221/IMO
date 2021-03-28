from typing import List, Optional
import numpy as np
from random import randrange, choice


def search_swap_vertices_in_cycle(matrix, cycle) -> (Optional[int], Optional[int]):
    size = len(cycle)
    it = np.arange(size)
    np.random.shuffle(it)
    for first in it:
        for second in it[::-1]:
            b = matrix[cycle[(first - 1)]][cycle[first]] + matrix[cycle[first]][cycle[(first + 1) % size]] \
                + matrix[cycle[(second - 1)]][cycle[second]] + matrix[cycle[second]][cycle[(second + 1) % size]]
            a = matrix[cycle[(first - 1)]][cycle[second]] + matrix[cycle[second]][cycle[(first + 1) % size]] \
                + matrix[cycle[(second - 1)]][cycle[first]] + matrix[cycle[first]][cycle[(second + 1) % size]]
            if a < b:
                if abs(first - second) == 1:
                    b = matrix[cycle[(first - 1)]][cycle[first]] + matrix[cycle[second]][cycle[(second + 1) % size]]
                    a = matrix[cycle[(first - 1)]][cycle[second]] + matrix[cycle[first]][cycle[(second + 1) % size]]
                    if a >= b:
                        continue
                elif ((first + 1) % size == second) or (second + 1) % size == first:
                    first, second = min(first, second), max(first, second)
                    b = matrix[cycle[(first + 1)]][cycle[first]] + matrix[cycle[second]][cycle[second - 1]]
                    a = matrix[cycle[(first + 1)]][cycle[second]] + matrix[cycle[first]][cycle[second - 1]]
                    if a >= b:
                        continue
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


def greedy_swap_edges_in_cycle(matrix,cycle1,cycle2):
    search = True
    while search:
        i, j = search_swap_edges_in_cycle_greedy(matrix, cycle1)
        
        if i is None or j is None:
           search = False
        else:
            cycle1[i + 1], cycle1[j] = cycle1[j], cycle1[i + 1]
            cycle1[i + 2:j] = cycle1[i + 2:j][::-1]
           
            
    search = True
    while search:
        i, j = search_swap_edges_in_cycle_greedy(matrix, cycle2)
        
        if i is None or j is None:
           search = False
        else:
            cycle2[i + 1], cycle2[j] = cycle2[j], cycle2[i + 1]
            cycle2[i + 2:j] = cycle2[i + 2:j][::-1]
           
    
    return cycle1,cycle2
            
      
def search_swap_edges_in_cycle_greedy(matrix, cycle):
    size = len(cycle) - 1
    it = np.arange(size)
    np.random.shuffle(it)
    for first in it:
        for second in it[::-1]:
            if abs(first - second) == 1:
                continue
            if (first == 0 or first == size - 1) and (second == 0 or second == size - 1): #edge case
                continue
            
            i = min(first,second) # i zawsze mniejsze od j (chodzi o indeksy)
            j = max(first,second)
            old_distance = matrix[cycle[i]][cycle[(i + 1) % size]] + matrix[cycle[j]][cycle[(j + 1) % size]]
            new_distance = matrix[cycle[i]][cycle[j]] + matrix[cycle[(i + 1) % size]][cycle[(j + 1) % size]]

            if new_distance - old_distance < 0:
                return i, j
            
    return None, None


