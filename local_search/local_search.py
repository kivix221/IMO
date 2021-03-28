try:
    from .greedy_search import *
    from .steepest_search import *
    from ..greedy_heuristics.regret_cycle import double_regret_cycle
except Exception:
    from IMO.local_search.steepest_search import *
    from IMO.local_search.greedy_search import *
    from IMO.greedy_heuristics.regret_cycle import double_regret_cycle
    
    # 2 podstawowe algorytmy:
# 1)Zamiana wierzchołków pomiędzy cyklami i w cyklach -- swap_vertices_in_cycle
# 2)Zamiana wierzchołków pomiędzy cyklami i krawędzi w cyklach -- swap_edges_in_cycle
# w 2 wariantach: greedy i steep

def local_greedy_swap_vertices_in_cycle(matrix, cycle1, cycle2):
    cycle1, cycle2 = greedy_swap_vertices_in_cycle(matrix, cycle1[:-1], cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    
    cycle1, cycle2 = greedy_swap_vertices_between_cycle(matrix, cycle1[:-1], cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    
    return cycle1, cycle2


def local_greedy_swap_edges_in_cycle(matrix, cycle1, cycle2):
    cycle1, cycle2 = greedy_swap_vertices_between_cycle(matrix, cycle1[:-1], cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    
    return greedy_swap_edges_in_cycle(matrix, cycle1, cycle2)


def local_steep_swap_vertices_in_cycle(matrix, cycle1, cycle2):
    cycle1, cycle2 = steep_swap_vertices_in_cycle(matrix, cycle1[:-1]), steep_swap_vertices_in_cycle(matrix,                                                                                             cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    
    cycle1, cycle2 = steep_swap_vertices_between_cycle(matrix, cycle1[:-1], cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    
    return cycle1,cycle2


def local_steep_swap_edges_in_cycle(matrix, cycle1, cycle2):
    cycle1, cycle2 = steep_swap_vertices_between_cycle(matrix, cycle1[:-1], cycle2[:-1])
    cycle1, cycle2 = np.concatenate([cycle1, [cycle1[0]]]), np.concatenate([cycle2, [cycle2[0]]])
    
    return steepest_swap_edges_in_cycle(matrix, cycle1, cycle2)