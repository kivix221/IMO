from utils import *
from local_search.local_search import local_greedy_swap_edges_in_cycle
from global_convexity.helpers import *

kroa200_instance = load_instance('./data/kroa200.tsp')
krob200_instance = load_instance('./data/krob200.tsp')
kroa200_distance_matrix = calc_distance_matrix(kroa200_instance)
krob200_distance_matrix = calc_distance_matrix(krob200_instance)



ITER = 1000
solutions = []
distances = []
for _ in range(ITER):
   cycle1, cycle2 = get_random_cycle(len(krob200_distance_matrix))
   
   solutions.append(local_greedy_swap_edges_in_cycle(krob200_distance_matrix,cycle1,cycle2))
   
   distances.append(get_cycles_distance(krob200_distance_matrix,
                                        *solutions[-1])[0])
   
   
best_solution = solutions[np.argmin(distances)]

similarity = calc_similarity_vs_best(solutions,best_solution) 
print(calc_corrcoef(distances,similarity))
plot_distance_vs_similarity(distances,similarity,"Podobienstwo-wierzchołki-krob200")

similarity = calc_similarity_vs_others(solutions) 
print(calc_corrcoef(distances,similarity))
plot_distance_vs_similarity(distances,similarity,"Średnie-podobienstwo-wierzchołki-krob200")


#####################################
edges_solution = [(get_edges(solution[0]),get_edges(solution[1])) for solution in solutions]
edges_best = (get_edges(best_solution[0]),get_edges(best_solution[1]))

similarity = calc_similarity_vs_best(edges_solution,edges_best,False) 
print(calc_corrcoef(distances,similarity))
plot_distance_vs_similarity(distances,similarity,"Podobienstwo-krawędzie-krob200")

similarity = calc_similarity_vs_others(edges_solution,False) 
print(calc_corrcoef(distances,similarity))
plot_distance_vs_similarity(distances,similarity,"Średnie-podobienstwo-krawędzie-krob200")
    
