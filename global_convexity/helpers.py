import numpy as np
import matplotlib.pyplot as plt



def calc_similarity_vs_best(solutions,best_solution,check_vert = True):
    similarity = [calc_similarity(solution,best_solution,check_vert) for solution in solutions]#bedzie sam siebie liczyl xD
    return similarity

def calc_similarity_vs_others(solutions,check_vert = True):
    similarity = []
    for i in range(len(solutions)):
        single_sim = [calc_similarity(solutions[i],solutions[j],check_vert) if i != j else 0 for j in range(len(solutions))] 
        similarity.append(np.sum(single_sim)/(len(solutions)-1 ))
    
    return similarity

### ULEPSZYĆ RAZ ZLICZAJĄC EDGE !!!

def calc_similarity(cycles,cycle_ref,check_vert = True):
    if check_vert:
        ref_c1_vert,ref_c2_vert = set(cycle_ref[0]), set(cycle_ref[1])
        
        shared_vert_c1 = max(len(list(ref_c1_vert.intersection(cycles[0]))),
                             len(list(ref_c2_vert.intersection(cycles[0]))))
        
        shared_vert_c2 = max(len(list(ref_c1_vert.intersection(cycles[1]))),
                             len(list(ref_c2_vert.intersection(cycles[1]))))
        
        return shared_vert_c1 + shared_vert_c2
    
    else:
       ref_c1_edges,ref_c2_edges = cycle_ref[0],cycle_ref[1]
       c1_edges, c2_edges =  cycles[0],cycles[1]
       
       shared_edges_c1 = max(calc_shared_edges(c1_edges,ref_c1_edges),
                            calc_shared_edges(c1_edges,ref_c2_edges))
       shared_edges_c2 = max(calc_shared_edges(c2_edges,ref_c1_edges),
                            calc_shared_edges(c2_edges,ref_c2_edges))
       
       return  shared_edges_c1 + shared_edges_c2
        

def get_edges(cycle):
    edges = [(cycle[i],cycle[(i+1) % len(cycle)])  for i in range(len(cycle))]
    return edges

def calc_shared_edges(cycle,cycle_ref):
    shared_edges = 0
    for i in range(len(cycle)):
            for j in range(len(cycle_ref)):
                #(a,b) = (c,d)
                is_exist = (cycle[i][0] == cycle_ref[j][0]) and (cycle[i][1] == cycle_ref[j][1])
                #(a,b) = (d,c) może być kolejność odwrócona wierzchołków
                is_exist_inverse = (cycle[i][0] == cycle_ref[j][1]) and (cycle[i][1] == cycle_ref[j][0])
            
                if is_exist or is_exist_inverse:
                  shared_edges += 1
    
    return shared_edges
    

def calc_corrcoef(distances,similarity):
    return np.corrcoef(distances,similarity)

def plot_distance_vs_similarity(distance,similarity,title):
    plt.figure()
    plt.scatter(distance,similarity,alpha=0.5)
    plt.ylim(bottom=0,top = 200)
    #plt.xlim(left = 0)
    plt.title(title)
    plt.xlabel("Funkcja celu")
    plt.ylabel("Podobieństwo")
    plt.legend()
    plt.savefig(title)

 

    
