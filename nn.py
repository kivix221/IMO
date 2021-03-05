import numpy as np
import random

def nn_alg(distance_matrix, node=0):
    first_cycle =  [random.randint(0,99)]
    second_cycle = [random.randint(0,99)]

    while len(first_cycle) < len(distance_matrix)/2:
        first_cycle_city = nearest_city(first_cycle + second_cycle, distance_matrix, first_cycle[len(first_cycle) - 1]) 
        first_cycle.append(first_cycle_city) 
     
        second_cycle_city = nearest_city(first_cycle + second_cycle, distance_matrix, second_cycle[len(second_cycle) - 1]) 
        if second_cycle_city != -1:
            second_cycle.append(second_cycle_city)

    first_cycle.append(first_cycle[0])
    second_cycle.append(second_cycle[0])

    return first_cycle, second_cycle

def nearest_city(used_cities,dist_matrix, current_city):
    max_dist = np.inf
    nearest_city = -1
    
    for i in range(len(dist_matrix[current_city])):
        if i in used_cities:
            continue

        if dist_matrix[current_city,i] < max_dist:
            max_dist = dist_matrix[current_city,i]
            nearest_city = i

    return nearest_city

