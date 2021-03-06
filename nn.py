import numpy as np
import random
from utils import get_cycle_distance


def nn_alg(distance_matrix, node = 0):
    not_been = list(range(len(distance_matrix)))

   
    cycle1_current_node = not_been.pop(random.randint(0,99))
    #drugi losowo
    cycle2_current_node = not_been.pop(random.randint(0,98))
    
    #drugi jak najdalej
    # farest_node = np.where(np.isinf(distance_matrix[cycle1_current_node]),
    #             -np.inf,distance_matrix[cycle1_current_node]).argmax()
    # cycle2_current_node = not_been.pop(not_been.index(farest_node))
                 
    cycle1 = [cycle1_current_node]
    cycle2 = [cycle2_current_node]

    while len(not_been) != 0:
        cycle1_new_node = get_nearest_node(distance_matrix,not_been, cycle1_current_node)
        cycle1_insert_index =  get_best_insertion(distance_matrix,cycle1,cycle1_new_node)
        cycle1.insert(cycle1_insert_index,cycle1_new_node)
        cycle1_current_node = cycle1_new_node

        cycle2_new_node = get_nearest_node(distance_matrix,not_been,cycle2_current_node)
        cycle2_insert_index = get_best_insertion(distance_matrix,cycle2,cycle2_new_node)
        cycle2.insert(cycle2_insert_index,cycle2_new_node)
        cycle2_current_node = cycle2_new_node  

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


def get_best_insertion(distance_matrix,cycle,new_node):
    if len(cycle) == 1:
        return 1

    min_total_distance = np.inf
    best_insertion = None

    for i in range(len(cycle) + 1):
        cycle.insert(i,new_node)
        total_distance = get_cycle_distance(distance_matrix,cycle)
        if(total_distance< min_total_distance):
            min_total_distance = total_distance
            best_insertion = i
        cycle.pop(i)

    return best_insertion


