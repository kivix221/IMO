# TO DO
import numpy as np

from candidates import steep_candidates, is_in_cycle, calculate_candidates
from utils import *
from time import time
from greedy_heuristics.regret_cycle import double_regret_cycle
from local_search_2.steep_lm import steep_lm
from local_search.local_search import steep_swap_vertices_between_cycle

def run_local(instance,name):
    res = []
    res_t = []
    min_res = np.inf
    min_cy = None
    
    matrix = calc_distance_matrix(instance)
  
    for _ in range(100):
        cy1, cy2 = get_random_cycle(len(matrix))
        start = time()
        cy1, cy2 = steep_swap_vertices_between_cycle(matrix,cy1,cy2)
        res_t.append(time() - start)
        res.append(get_cycles_distance(matrix, cy1, cy2)[0])
        if res[-1] < min_res:
            min_res = res[-1]
            min_cy = (cy1, cy2)
        print(_, end='|')

    print("\n========================")
    print(name)
    plot_result(instance, min_cy[0], min_cy[1], name)
    print(np.mean(res), np.min(res), np.max(res))
    print(np.mean(res_t), np.min(res_t), np.max(res_t))

def run_steep_lm(instance, name):
    res = []
    res_t = []
    min_res = np.inf
    min_cy = None
    
    matrix = calc_distance_matrix(instance)
  
    for _ in range(100):
        cy1, cy2 = get_random_cycle(len(matrix))
        start = time()
        cy1, cy2 = steep_lm(matrix,cy1,cy2)
        res_t.append(time() - start)
        res.append(get_cycles_distance(matrix, cy1, cy2)[0])
        if res[-1] < min_res:
            min_res = res[-1]
            min_cy = (cy1, cy2)
        print(_, end='|')

    print("\n========================")
    print(name)
    plot_result(instance, min_cy[0], min_cy[1], name)
    print(np.mean(res), np.min(res), np.max(res))
    print(np.mean(res_t), np.min(res_t), np.max(res_t))

def run_candidates(instance, name):
    res = []
    res_t = []
    min_res = np.inf
    min_cy = None

    matrix = calc_distance_matrix(instance)
    candidates = calculate_candidates(matrix)

    for _ in range(100):
        cy1, cy2 = get_random_cycle(len(matrix))
        start = time()
        cy1, cy2 = steep_candidates(matrix, cy1, cy2, candidates, is_in_cycle(len(matrix), cy1, cy2))
        res_t.append(time() - start)
        res.append(get_cycles_distance(matrix, cy1, cy2)[0])
        if res[-1] < min_res:
            min_res = res[-1]
            min_cy = (cy1, cy2)
        print(_, end='|')

    print("\n========================")
    print(name)
    plot_result(instance, min_cy[0], min_cy[1], name)
    print(np.mean(res), np.min(res), np.max(res))
    print(np.mean(res_t), np.min(res_t), np.max(res_t))


def run_regret(instance, name):
    res = []
    res_t = []
    min_res = np.inf
    min_cy = None

    matrix = calc_distance_matrix(instance)

    for _ in range(100):
        start = time()
        cy1, cy2 = double_regret_cycle(matrix, 2*_)
        res_t.append(time() - start)
        res.append(get_cycles_distance(matrix, cy1, cy2)[0])
        if res[-1] < min_res:
            min_res = res[-1]
            min_cy = (cy1, cy2)
        print(_, end='|')

    print("\n========================")
    print(name)
    plot_result(instance, min_cy[0], min_cy[1], name)
    print(np.mean(res), np.min(res), np.max(res))
    print(np.mean(res_t), np.min(res_t), np.max(res_t))


if __name__ == "__main__":
    ka200_instance = load_instance('../data/kroa200.tsp')
    kb200_instance = load_instance('../data/krob200.tsp')

    run_local(ka200_instance, 'regret_cycles')
    run_local(kb200_instance, 'regret_cycles')
    
    run_steep_lm(ka200_instance, 'regret_cycles')
    run_steep_lm(kb200_instance, 'regret_cycles')
    
    run_candidates(ka200_instance, 'regret_cycles')
    run_candidates(kb200_instance, 'regret_cycles')

    run_regret(ka200_instance, 'regret_cycles')
    run_regret(kb200_instance, 'regret_cycles')
    
    
