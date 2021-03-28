import numpy as np
from random import choice

def random_wandering(matrix,cycle1,cycle2):
    for _ in range(3*10**4):
        f, s = random_swap_vertices_between_cycle(matrix, cycle1, cycle2)
        cycle1[f], cycle2[s] = cycle2[s], cycle1[f]
        
        if f == 0:
            cycle1[-1] = cycle1[0]
        if s == 0:
            cycle2[-1] = cycle2[0]
        if f == len(cycle1) -1:
            cycle1[0] = cycle1[-1]
        if s == len(cycle2) - 1:
            cycle2[0] = cycle2[-1]
            
        if choice([True, False]):
            f,s = random_swap_vertices_in_cycle(matrix,cycle1)
            cycle1[f], cycle1[s] = cycle1[s], cycle1[f]
            if f == 0 or s == 0:
                cycle1[-1] = cycle1[0]
            if f == len(cycle1) -1 or s == len(cycle1) -1:
                cycle1[0] = cycle1[-1]
            
            f,s = random_swap_vertices_in_cycle(matrix,cycle2)
            cycle2[f], cycle2[s] = cycle2[s], cycle2[f]
            if f == 0 or s == 0:
                cycle2[-1] = cycle2[0]
            if f == len(cycle2) -1 or s == len(cycle2) -1:
                cycle2[0] = cycle2[-1]
          
        else:
            f,s = random_swap_edges_in_cycle_greedy(matrix,cycle1)
            cycle1[f + 1], cycle1[s] = cycle1[s], cycle1[f + 1]
            cycle1[f + 2:s] = cycle1[f + 2:s][::-1]
            
            f,s = random_swap_edges_in_cycle_greedy(matrix,cycle2)
            cycle2[f + 1], cycle2[s] = cycle2[s], cycle2[f + 1]
            cycle2[f + 2:s] = cycle2[f + 2:s][::-1]
            
    return cycle1,cycle2


def random_swap_vertices_between_cycle(matrix, cycle1, cycle2):
    size1, size2 = len(cycle1), len(cycle2)
    it1, it2 = np.arange(size1), np.arange(size2)

    np.random.shuffle(it1)
    np.random.shuffle(it2)
    if choice([True, False]):
        for first in it1:
            for second in it2:     
                return first, second
    else:
        for second in it2:
            for first in it1:
                return first, second
            
def random_swap_vertices_in_cycle(matrix, cycle):
    size = len(cycle)
    it = np.arange(size)
    np.random.shuffle(it)
    for first in it:
        for second in it[::-1]:
            b = matrix[cycle[(first - 1)]][cycle[first]] + matrix[cycle[first]][cycle[(first + 1) % size]] \
                + matrix[cycle[(second - 1)]][cycle[second]] + matrix[cycle[second]][cycle[(second + 1) % size]]
            a = matrix[cycle[(first - 1)]][cycle[second]] + matrix[cycle[second]][cycle[(first + 1) % size]] \
                + matrix[cycle[(second - 1)]][cycle[first]] + matrix[cycle[first]][cycle[(second + 1) % size]]

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

def random_swap_edges_in_cycle_greedy(matrix, cycle):
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
        
            return i, j
            

