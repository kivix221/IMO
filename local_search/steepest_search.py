import numpy as np
from ..utils import get_random_cycle, load_instance, get_cycles_distance, calc_distance_matrix,plot_result

#Wersje steepest algorytmow

#1)z.w + 2)z.k
def steepest_swap_edges_in_cycle(matrix,cycle1,cycle2):
    search = True
    while search:
        #albo pomiędzy albo cykl1 albo cykl2 -- zastanawiam się czy węwnatransowe jednocześnie?
        best_movement = None # 1 - z.w; 2 - z.k w cyklu1; 3 - z.k w cyklu2
        best_delta = np.inf
        best_i, best_j = None, None

        # TO DO
        # zamiana wierzchołków pomiędzy cyklami steepest 
       
        #zamiana krawędzi cykl1
        delta,i,j= search_swap_edges_in_cycle_steepest(matrix,cycle1)
        if delta < best_delta:
            best_movement = 2
            best_delta = delta
            best_i,best_j = i,j
        
        delta,i,j = search_swap_edges_in_cycle_steepest(matrix,cycle2)
        if delta < best_delta:
            best_movement = 3
            best_delta = delta
            best_i,best_j = i,j
        
        #Zrobienie najlepszego ruchu
        if best_movement == 1:
            cycle1[best_i], cycle2[best_j] = cycle2[best_j], cycle1[best_i]
        elif best_movement == 2:
            cycle1[best_i+1], cycle1[best_j] = cycle1[best_j], cycle1[best_i+1]
            cycle1[best_i + 2:best_j] = cycle1[best_i + 2:j][::-1]
        elif best_movement == 3:
            cycle2[best_i+1], cycle2[best_j] = cycle2[best_j], cycle2[best_i+1]
            cycle2[best_i + 2:best_j] = cycle2[best_i + 2:j][::-1]
        
        #przerwanie
        if best_movement is None:
            search = False

    return cycle1,cycle2

def search_swap_edges_in_cycle_steepest(matrix,cycle):
    best_delta = np.inf
    best_i, best_j = None, None
    for i in range(len(cycle) - 1):
        for j in range(i +2, len(cycle) - 1):
            old_distance = matrix[cycle[i]][cycle[i+1]] + matrix[cycle[j]][cycle[j+1]]
            new_distance = matrix[cycle[i]][cycle[j]] + matrix[cycle[i+1]][cycle[j+1]]

            delta = new_distance - old_distance
            if delta < best_delta:
               best_delta = delta
               best_i,best_j = i,j

    return best_delta,best_i,best_j