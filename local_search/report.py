try:
    from .local_search import *
    from .random_wandering import random_wandering
    from ..utils import *
except Exception:
    from IMO.local_search.local_search import *
    from IMO.local_search.random_wandering import *
    from IMO.utils import *
    
import pandas as pd
import time
  
## !!! do sterowania
func_seq = (random_wandering,local_greedy_swap_vertices_in_cycle,local_greedy_swap_edges_in_cycle,
            local_steep_swap_vertices_in_cycle,local_steep_swap_edges_in_cycle) 
name_func = ("random_wandering","local_greedy_swap_vertices_in_cycle", "local_greedy_swap_edges_in_cycle",
             "local_steep_swap_vertices_in_cycle","local_steep_swap_edges_in_cycle") 
REPEAT = 100
## !!! do sterowania

## LEGENDA!!! ####
# d[name][inx][0] - srednia dist
# d[name][inx][1] - min dist
# d[name][inx][2] - max dist
# d[name][inx][3] - sredni t
# d[name[inx][4] - min t
# d[name][inx][5] - max t
# d[name][inx][6] - best cycle 1, best cycle2

def call_search(matrix, d, cycle1, cycle2, inx=0):
    for name, func in zip(name_func, func_seq):
        duration = time.time()
        tc1, tc2 = func(matrix, cycle1.copy(), cycle2.copy())
        duration = time.time() - duration
        t = get_cycles_distance(matrix, tc1, tc2)[0]
        d[name][inx][0] += t
        d[name][inx][1] = min(d[name][inx][1], t)
        d[name][inx][2] = max(d[name][inx][2], t)
        d[name][inx][3] += duration
        d[name][inx][4] = min(d[name][inx][4],duration)
        d[name][inx][5] = max(d[name][inx][5],duration)
        
        if d[name][inx][1] == t:
            d[name][inx][6] = tc1,tc2 

    return d


def test_random_start(instance1,instance2,matrix1, matrix2):
    d = dict()
    for n in name_func: d[n] = [[0, np.inf, -np.inf, 0., np.inf, -np.inf, (None,None)],\
        [0, np.inf, -np.inf, 0., np.inf, -np.inf,(None,None)]]
    for _ in range(REPEAT):
        cycle1, cycle2 = get_random_cycle(len(matrix1)) 
        d = call_search(matrix1, d, cycle1, cycle2)
        cycle1, cycle2 = get_random_cycle(len(matrix2))
        d = call_search(matrix2, d, cycle1, cycle2, 1)
        print(f'{_}|', end='')
    print('\n=============================================')
    for k in d.keys():
        d[k][0][0] /= REPEAT
        d[k][1][0] /= REPEAT
        d[k][0][3] /= REPEAT
        d[k][1][3] /= REPEAT
        
        for i in range(6): #zaokrąglanie wyników
            d[k][0][i] = np.round(d[k][0][i])
            d[k][1][i] = np.round(d[k][1][i])
            
        plot_result(instance1,*d[k][0][6], k)
        plot_result(instance2,*d[k][1][6], k)
        
        del d[k][0][6]
        del d[k][1][6]
    
    res = pd.DataFrame(data=d, index=('kroa100', 'krob100'))
    res.to_csv('result_random.csv')
    print(res)
    
   
def test_heuristic_start(instance1,instance2,matrix1, matrix2, algorithm):
    d = dict()
    for n in name_func: d[n] = [[0, np.inf, -np.inf, 0., np.inf, -np.inf, (None,None)],\
        [0, np.inf, -np.inf, 0., np.inf, -np.inf,(None,None)]]
    for _ in range(REPEAT):
        cycle1, cycle2 = algorithm(matrix1, node=_)
        d = call_search(matrix1, d, cycle1, cycle2)
        cycle1, cycle2 = algorithm(matrix2, node=_)
        d = call_search(matrix2, d, cycle1, cycle2, 1)
        print(f'{_}|', end='')
    print('\n=============================================')
    for k in d.keys():
        d[k][0][0] /= REPEAT
        d[k][1][0] /= REPEAT
        d[k][0][3] /= REPEAT
        d[k][1][3] /= REPEAT
        
        for i in range(6): #zaokrąglanie wyników
            d[k][0][i] = np.round(d[k][0][i])
            d[k][1][i] = np.round(d[k][1][i])
            
        plot_result(instance1,*d[k][0][6], k)
        plot_result(instance2,*d[k][1][6], k)
        
        del d[k][0][6]
        del d[k][1][6]
    
    res = pd.DataFrame(data=d, index=('kroa100', 'krob100'))
    res.to_csv('result_heuristic.csv')
    print(res)   


if __name__ == "__main__":
    kroa100_instance = load_instance('../data/kroa100.tsp')
    krob100_instance = load_instance('../data/krob100.tsp')

    kroa100_distance_matrix = calc_distance_matrix(kroa100_instance)
    krob100_distance_matrix = calc_distance_matrix(krob100_instance)

    test_random_start(kroa100_instance,krob100_instance,kroa100_distance_matrix, krob100_distance_matrix)
    test_heuristic_start(kroa100_instance,krob100_instance,kroa100_distance_matrix, krob100_distance_matrix, double_regret_cycle)
