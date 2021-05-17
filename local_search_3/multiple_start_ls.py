import random
import numpy as np
from local_search_2.candidates import steep_candidates, calculate_candidates, is_in_cycle
from utils import get_cycles_distance
from greedy_heuristics.regret_cycle import double_regret_cycle

def multiple_start_local_search(matrix):
    best_distance = np.Infinity
    best_c1, best_c2 = [],[]
    candidates = calculate_candidates(matrix)

    for _ in range(100):
        cycle1, cycle2 =  double_regret_cycle(matrix, random.randrange(len(matrix)))
  
        cycle1, cycle2 = steep_candidates(matrix,cycle1,cycle2,candidates,
                                            is_in_cycle(len(matrix),cycle1,cycle2))
        cycles_distance,_,__ = get_cycles_distance(matrix,cycle1,cycle2)
        if cycles_distance < best_distance:
            best_distance = cycles_distance
            best_c1,best_c2 = cycle1,cycle2
            
    return best_c1,best_c2
  