import numpy as np
import time

try:
    from ..utils import get_random_cycle, load_instance, get_cycles_distance, calc_distance_matrix, plot_result
except Exception:
    from IMO.utils import get_random_cycle, load_instance, get_cycles_distance, calc_distance_matrix, plot_result


# Wersje steepest algorytmow

# 1)z.w + 2)z.k
def steepest_swap_edges_in_cycle(matrix, cycle1, cycle2):
    search = True
    while search:
        # albo pomiędzy albo cykl1 albo cykl2 -- zastanawiam się czy węwnatransowe jednocześnie?
        best_movement = None  # 1 - z.w; 2 - z.k w cyklu1; 3 - z.k w cyklu2
        best_delta = np.inf
        best_i, best_j = None, None

        # TO DO
        # zamiana wierzchołków pomiędzy cyklami steepest 

        # zamiana krawędzi cykl1
        delta, i, j = search_swap_edges_in_cycle_steepest(matrix, cycle1)
        if delta < best_delta:
            best_movement = 2
            best_delta = delta
            best_i, best_j = i, j

        delta, i, j = search_swap_edges_in_cycle_steepest(matrix, cycle2)
        if delta < best_delta:
            best_movement = 3
            best_delta = delta
            best_i, best_j = i, j

        # Zrobienie najlepszego ruchu
        if best_movement == 1:
            cycle1[best_i], cycle2[best_j] = cycle2[best_j], cycle1[best_i]
        elif best_movement == 2:
            cycle1[best_i + 1], cycle1[best_j] = cycle1[best_j], cycle1[best_i + 1]
            cycle1[best_i + 2:best_j] = cycle1[best_i + 2:j][::-1]
        elif best_movement == 3:
            cycle2[best_i + 1], cycle2[best_j] = cycle2[best_j], cycle2[best_i + 1]
            cycle2[best_i + 2:best_j] = cycle2[best_i + 2:j][::-1]

        # przerwanie
        if best_movement is None:
            search = False

    return cycle1, cycle2


def search_swap_edges_in_cycle_steepest(matrix, cycle):
    best_delta = np.inf
    best_i, best_j = None, None
    for i in range(len(cycle) - 1):
        for j in range(i + 2, len(cycle) - 1):
            old_distance = matrix[cycle[i]][cycle[i + 1]] + matrix[cycle[j]][cycle[j + 1]]
            new_distance = matrix[cycle[i]][cycle[j]] + matrix[cycle[i + 1]][cycle[j + 1]]

            delta = new_distance - old_distance
            if delta < best_delta:
                best_delta = delta
                best_i, best_j = i, j

    return best_delta, best_i, best_j


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


if __name__ == "__main__":
    kroa100_instance = load_instance('../data/kroa100.tsp')
    kroa100_distance_matrix = calc_distance_matrix(kroa100_instance)
    for i in range(len(kroa100_distance_matrix)):
        kroa100_distance_matrix[i, i] = 0

    k_cycle1, k_cycle2 = get_random_cycle()
    print("Pierwotnie: ", get_cycles_distance(kroa100_distance_matrix, k_cycle1, k_cycle2))
    plot_result(kroa100_instance, k_cycle1, k_cycle2)

    v_cycle1, v_cycle2 = k_cycle1.copy(), k_cycle2.copy()  # do greedy swap vert in cycle
    v2_cycle1, v2_cycle2 = k_cycle1.copy(), k_cycle2.copy()  # do greedy swap vert beetween cycle
    v3_cycle1, v3_cycle2 = k_cycle1.copy(), k_cycle2.copy()  # do greedy swap edges

    # swap vertices
    print(len(v_cycle1), len(v_cycle2))
    duration = time.time()
    v_cycle1 = steep_swap_vertices_in_cycle(kroa100_distance_matrix, v_cycle1[:-1])
    v_cycle2 = steep_swap_vertices_in_cycle(kroa100_distance_matrix, v_cycle2[:-1])
    v_cycle1 = np.concatenate([v_cycle1, [v_cycle1[0]]])
    v_cycle2 = np.concatenate([v_cycle2, [v_cycle2[0]]])
    duration = time.time() - duration
    print(len(v_cycle1), len(v_cycle2))
    print("Steep swap vert in cycle: ", get_cycles_distance(kroa100_distance_matrix, v_cycle1, v_cycle2))
    print("Steep swap vert in cycle: ", duration)
    plot_result(kroa100_instance, v_cycle1, v_cycle2)

    duration = time.time()
    v2_cycle1, v2_cycle2 = steep_swap_vertices_between_cycle(kroa100_distance_matrix, v2_cycle1[:-1], v2_cycle2[:-1])
    v2_cycle1 = np.concatenate([v2_cycle1, [v2_cycle1[0]]])
    v2_cycle2 = np.concatenate([v2_cycle2, [v2_cycle2[0]]])
    duration = time.time() - duration
    print("Greedy swap vert between cycles: ", get_cycles_distance(kroa100_distance_matrix, v2_cycle1, v2_cycle2))
    print("Greedy swap vert between cycles: ", duration)
    plot_result(kroa100_instance, v2_cycle1, v2_cycle2)
