import numpy as np

# Wersje steepest algorytmow

def steepest_swap_edges_in_cycle(matrix, cycle1, cycle2):
    search = True
    while search:
        i, j = search_swap_edges_in_cycle_steepest(matrix, cycle1)
        
        if i is None or j is None:
           search = False
        else:
            cycle1[i + 1], cycle1[j] = cycle1[j], cycle1[i + 1]
            cycle1[i + 2:j] = cycle1[i + 2:j][::-1]
           
            
    search = True
    while search:
        i, j = search_swap_edges_in_cycle_steepest(matrix, cycle2)
        
        if i is None or j is None:
           search = False
        else:
            cycle2[i + 1], cycle2[j] = cycle2[j], cycle2[i + 1]
            cycle2[i + 2:j] = cycle2[i + 2:j][::-1]
           
    
    return cycle1,cycle2


def search_swap_edges_in_cycle_steepest(matrix, cycle):
    best_i, best_j = None, None
    for i in range(len(cycle) - 1):
        for j in range(i + 2, len(cycle) - 1):
            old_distance = matrix[cycle[i]][cycle[i + 1]] + matrix[cycle[j]][cycle[j + 1]]
            new_distance = matrix[cycle[i]][cycle[j]] + matrix[cycle[i + 1]][cycle[j + 1]]

            delta = new_distance - old_distance
            if new_distance - old_distance < 0:
                best_i, best_j = i, j

    return  best_i, best_j


def steep_swap_vertices_in_cycle(matrix, cycle):
    n = len(cycle)
    best_i, best_j = -1, -1

    while True:
        bd = 0
        for i, c in enumerate(cycle[:-1]):
            for jp, d in enumerate(cycle[(i + 1):]):
                j = jp + i + 1
                tb = matrix[cycle[i - 1]][c] + matrix[c][cycle[(i + 1) % n]] + matrix[cycle[j - 1]][d] + matrix[d][
                    cycle[(j + 1) % n]]
                ta = matrix[cycle[i - 1]][d] + matrix[d][cycle[(i + 1) % n]] + matrix[cycle[j - 1]][c] + matrix[c][
                    cycle[(j + 1) % n]]
                if ta - tb < bd:
                    if i + 1 == j:
                        tb = matrix[cycle[i - 1]][c] + matrix[d][cycle[(j + 1) % n]]
                        ta = matrix[cycle[i - 1]][d] + matrix[c][cycle[(j + 1) % n]]
                        if ta - tb < bd:
                            bd = ta - tb
                            best_i, best_j = i, j
                    elif i == (j + 1) % n:
                        tb = matrix[cycle[i + 1]][c] + matrix[d][cycle[j - 1]]
                        ta = matrix[cycle[i + 1]][d] + matrix[c][cycle[j - 1]]
                        if ta - tb < bd:
                            bd = ta - tb
                            best_i, best_j = i, j
                    else:
                        bd = ta - tb
                        best_i, best_j = i, j
        if bd < 0:
            cycle[best_i], cycle[best_j] = cycle[best_j], cycle[best_i]

        else:
            break
    return cycle


def steep_swap_vertices_between_cycle(matrix, cycle1, cycle2):
    n1, n2 = len(cycle1), len(cycle2)
    best_i, best_j = -1, -1

    while True:
        bd = 0
        for i, c in enumerate(cycle1):
            for j, d in enumerate(cycle2):
                tb = matrix[cycle1[i - 1]][c] + matrix[c][cycle1[(i + 1) % n1]] + matrix[cycle2[j - 1]][d] + matrix[d][
                    cycle2[(j + 1) % n2]]
                ta = matrix[cycle1[i - 1]][d] + matrix[d][cycle1[(i + 1) % n1]] + matrix[cycle2[j - 1]][c] + matrix[c][
                    cycle2[(j + 1) % n2]]
                if ta - tb < bd:
                    bd = ta - tb
                    best_i, best_j = i, j
        if bd < 0:
            cycle1[best_i], cycle2[best_j] = cycle2[best_j], cycle1[best_i]

        else:
            break
    return cycle1, cycle2
