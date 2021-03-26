try:
    from .greedy_search import *
    from .steepest_search import *
    from ..greedy_heuristics.regret_cycle import double_regret_cycle
except Exception:
    from IMO.local_search.steepest_search import *
    from IMO.local_search.greedy_search import *
    from IMO.greedy_heuristics.regret_cycle import double_regret_cycle
import pandas as pd


def greedy_in_cycle(matrix, cycle1, cycle2):
    cycle1, cycle2 = greedy_swap_vertices_in_cycle(matrix, cycle1[:-1], cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    return greedy_swap_edges_in_cycle(matrix, cycle1, cycle2)


def greedy_between_cycle(matrix, cycle1, cycle2):
    cycle1, cycle2 = greedy_swap_vertices_between_cycle(matrix, cycle1[:-1], cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    return greedy_swap_edges_in_cycle(matrix, cycle1, cycle2)


def steep_in_cycle(matrix, cycle1, cycle2):
    cycle1, cycle2 = steep_swap_vertices_in_cycle(matrix, cycle1[:-1]), steep_swap_vertices_in_cycle(matrix,
                                                                                                     cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    return steepest_swap_edges_in_cycle(matrix, cycle1, cycle2)


def steep_between_cycle(matrix, cycle1, cycle2):
    cycle1, cycle2 = steep_swap_vertices_between_cycle(matrix, cycle1[:-1], cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    return steepest_swap_edges_in_cycle(matrix, cycle1, cycle2)


## !!! do sterowania
func_seq = (greedy_in_cycle, greedy_between_cycle)  #, steep_in_cycle, steep_between_cycle)
name_func = ("greedy_in_cycle", "greedy_between_cycle")  #, "steep_in_cycle", "steep_between_cycle")
REPEAT = 10
## !!! do sterowania


def call_search(matrix, d, cycle1, cycle2, inx=0):
    for name, func in zip(name_func, func_seq):
        tc1, tc2 = func(matrix, cycle1.copy(), cycle2.copy())
        t = get_cycles_distance(matrix, tc1, tc2)[0]
        d[name][inx][0] += t
        d[name][inx][1] = min(d[name][0][1], t)
        d[name][inx][2] = max(d[name][0][2], t)
    return d


def test_random_start(matrix1, matrix2):
    d = dict()
    for n in name_func: d[n] = [[0, np.inf, -np.inf], [0, np.inf, -np.inf]]
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
    res = pd.DataFrame(data=d, index=('kroa100', 'krob100'))
    print(res)


def test_heuristic_start(matrix1, matrix2, algorithm):
    d = dict()
    for n in name_func: d[n] = [[0, np.inf, -np.inf], [0, np.inf, -np.inf]]
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
    res = pd.DataFrame(data=d, index=('kroa100', 'krob100'))
    print(res)


# TO DO raport końcowy do zrobienia tabelki
if __name__ == "__main__":
    kroa100_instance = load_instance('../data/kroa100.tsp')
    krob100_instance = load_instance('../data/krob100.tsp')

    kroa100_distance_matrix = calc_distance_matrix(kroa100_instance)
    krob100_distance_matrix = calc_distance_matrix(krob100_instance)

    test_random_start(kroa100_distance_matrix, krob100_distance_matrix)
    test_heuristic_start(kroa100_distance_matrix, krob100_distance_matrix, double_regret_cycle)
