import random
from time import time
from random import randrange

try:
    from ..greedy_heuristics.regret_cycle import get_next_regret_node, get_cycle_next_node, double_regret_cycle
    from ..utils import *
    from ..local_search_2.candidates import steep_candidates, calculate_candidates, is_in_cycle
except Exception:
    from utils import *
    from local_search_2.candidates import calculate_candidates, steep_candidates, is_in_cycle
    from greedy_heuristics.regret_cycle import get_next_regret_node, get_cycle_next_node, double_regret_cycle
    from local_search_2.candidates import steep_candidates, calculate_candidates, is_in_cycle
    
   
def genetic_local_search(matrix, stop_time,full = True):
    """
    Zgodnie z wykładem:
    X - populacja jako krotka ([cykl1,cykl2], długość)
    y - nowe rozwiązanie (potomek)
    =====
    Schemat działania:
    1) Wygeneruj populację początkową X (heurestyka + local search na niej)
    2) Rekombinacja 2 losowo wybranych rodziców z X (zgodnie z opisem z instrukcji do labów)
    3) Poprawa rozwiązania poprzez local search
    4) dodanie do populacji jak nowe rozwiązanie jest lepsze od najgorszego rozwiązania w populacji i nie powtarza się
    
    Algorytm działa w czasie średniego działania MLS (czas z poprzedniego labu)
    """
    candidates = calculate_candidates(matrix)
    X = generate_start_population(matrix,candidates) #[double_regret_cycle(matrix, random.randrange(len(matrix))) for _ in range(20)]
    
      
    t, t_dur = time(), 0.0
    while t_dur < stop_time:
        parent1,parent2 = random.sample(X,2)
        parent1, parent2 = parent1[0],parent2[0]
        
        y = recombination(parent1,parent2,matrix)
        
        if full: #wersja HAE (LP po rekombinacji) albo HAEa (bez lokalnego przeszukiwania po rekombinacji)
            y = steep_candidates(matrix,*y,candidates,
                                            is_in_cycle(len(matrix),*y))
        
        y_distance,_,__ = get_cycles_distance(matrix,*y)    
        x_max_distance, x_max_index = np.max([x[1] for x in X ]), np.argmax([x[1] for x in X ])
        
        if y_distance < x_max_distance and y_distance not in [x[1] for x in X ]:
            X[x_max_index] = (y,y_distance)
        
        t_dur = time() - t
    
    x_best = X[np.argmin([x[1] for x in X ])] #[0] należy dodać jeżeli chcemy tylko cykle dostać
    
    #do debugowania
    print("Populacja:",[x[1] for x in X ])
    return x_best

################

def generate_start_population(matrix,candidates):
    X = []
    X_distance = []
    while len(X) <= 20:
        x = double_regret_cycle(matrix, random.randrange(len(matrix)))
        x = steep_candidates(matrix,*x,candidates,
                                            is_in_cycle(len(matrix),*x))
        x_distance,_,__ = get_cycles_distance(matrix,*x)

        if x_distance not in X_distance:
            X_distance.append(x_distance)
            X.append(x) 

    X = [(x, x_distance) for x, x_distance in zip(X,X_distance)]
    
    return X

def recombination(parent1,parent2,matrix):
    parent1_c1,parent1_c2 = parent1[0][:-1],parent1[1][:-1]
    parent2_c1,parent2_c2 = parent2[0][:-1],parent2[1][:-1]
    
    parent1_edges = np.array([get_edges(parent1_c1),get_edges(parent1_c2)])
    parent1_edges = parent1_edges.reshape((parent1_edges.shape[0] * parent1_edges.shape[1],2))
    
    parent2_edges = np.array([get_edges(parent2_c1),get_edges(parent2_c2)])
    parent2_edges = parent2_edges.reshape((parent2_edges.shape[0] * parent2_edges.shape[1],2))
    
    y,removed = destroy(parent1_c1,parent1_c2,parent1_edges,parent2_edges)
    
    y = repair(*y,matrix,removed)
    
    return y
    
def repair(cycle1, cycle2, matrix, removed):
        s2 = matrix.shape[0] // 2
        s1 = s2 + matrix.shape[0] % 2

        c1, c2 = np.array(cycle1), np.array(cycle2)

        while c1.shape[0] < s1 and c2.shape[0] < s2:
            removed_ind, cycle_ind, cyc = get_next_regret_node(matrix, removed, c1, c2)
            if not cyc:
                c1 = np.insert(c1, cycle_ind + 1, removed[removed_ind])
            else:
                c2 = np.insert(c2, cycle_ind + 1, removed[removed_ind])
            removed = np.delete(removed, removed_ind)

        while c1.shape[0] < s1:
            r_ind, cyc_ind = get_cycle_next_node(matrix, removed, c1)
            c1 = np.insert(c1, cyc_ind + 1, removed[r_ind])
            removed = np.delete(removed, r_ind)

        while c2.shape[0] < s2:
            r_ind, cyc_ind = get_cycle_next_node(matrix, removed, c2)
            c2 = np.insert(c2, cyc_ind + 1, removed[r_ind])
            removed = np.delete(removed, r_ind)

        c1, c2 = np.append(c1, [c1[0]]), np.append(c2, [c2[0]])
        return (c1, c2)
    
def destroy(parent1_c1,parent1_c2,parent1_edges,parent2_edges):
    shared_edges = []
    for i in range(len(parent1_edges)):
        for j in range(len(parent2_edges)):
            #(a,b) = (c,d)
            is_exist = (parent1_edges[i][0] == parent2_edges[j][0]) and (parent1_edges[i][1] == parent2_edges[j][1])
            #(a,b) = (d,c) może być kolejność odwrócona wierzchołków
            is_exist_inverse = (parent1_edges[i][0] == parent2_edges[j][1]) and (parent1_edges[i][1] == parent2_edges[j][0])
        
            if is_exist or is_exist_inverse:
                shared_edges.append(parent1_edges[i])

    new_c1, new_c2 = [],[]
    
    for i in range(len(parent1_c1)):
        for j in range(len(shared_edges)):
            if parent1_c1[i] == shared_edges[j][0] and parent1_c1[(i+1) % len(parent1_c1)] == shared_edges[j][1]:
                new_c1.append(parent1_c1[i])
        
    for i in range(len(parent1_c2)):
        for j in range(len(shared_edges)):
            if parent1_c2[i] == shared_edges[j][0] and parent1_c2[(i+1) % len(parent1_c2)] == shared_edges[j][1]:
                new_c2.append(parent1_c2[i])

   
    c1_rm = np.array([node for node in parent1_c1 if node not in new_c1])
    c2_rm = np.array( [node for node in parent1_c2 if node not in new_c2])
    removed = np.concatenate((c1_rm,c2_rm))
    
    return (new_c1, new_c2), np.ravel(removed)
    
    
def get_edges(cycle):
    edges = [(cycle[i],cycle[(i+1) % len(cycle)])  for i in range(len(cycle))]
    return edges

