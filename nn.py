import numpy as np
import random
from utils import get_cycle_distance


def nn_alg(distance_matrix, node = 0):
    not_been = list(range(len(distance_matrix)))

    cycle1_node = not_been.pop(random.randint(0,99))
    cycle2_node = not_been.pop(random.randint(0,98))
           
    cycle1 = [cycle1_node]
    cycle2 = [cycle2_node]

    while len(not_been) != 0:
        cycle1_node = get_nearest_node(distance_matrix,not_been, cycle1_node)
        cycle1_insert_index = find_best_insertion(distance_matrix,cycle1,cycle1_node)
        cycle1.insert(cycle1_insert_index, cycle1_node)

        cycle2_node = get_nearest_node(distance_matrix,not_been, cycle2_node)
        cycle2_insert_index = find_best_insertion(distance_matrix,cycle2,cycle2_node)
        cycle2.insert(cycle2_insert_index, cycle2_node)
      
    cycle1.append(cycle1[0])
    cycle2.append(cycle2[0])

    return cycle1, cycle2

def get_nearest_node(distance_matrix,not_been, current_node):
    min_dist = np.inf
    nearest_node = None

    for index_not_been, node_not_been in enumerate(not_been):
       if distance_matrix[current_node, node_not_been] < min_dist:
            min_dist = distance_matrix[current_node, node_not_been]
            nearest_node = index_not_been

    return not_been.pop(nearest_node)

def find_best_insertion(distance_matrix,cycle,new_node):
    if len(cycle) == 1:
        return 1

    min_total_distance = np.inf
    best_insertion = None

    for position in range(len(cycle) + 1):
        cycle.insert(position,new_node)
        total_distance = get_cycle_distance(distance_matrix,cycle)
        if(total_distance< min_total_distance):
            min_total_distance = total_distance
            best_insertion = position
            
        cycle.pop(position)

    return best_insertion

