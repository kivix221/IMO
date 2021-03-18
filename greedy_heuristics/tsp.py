from IMO.greedy_heuristics.sandbox import double_greedy_cycle_seq, double_regret_cycle2
from IMO.utils import load_instance, calc_distance_matrix, plot_result, get_cycles_distance
from IMO.greedy_heuristics.nn import nn_alg
from IMO.greedy_heuristics.greedy_cycle import *
 
##########
kroa100_instance = load_instance('kroa100.tsp')
krob100_instance = load_instance('krob100.tsp')

kroa100_distance_matrix = calc_distance_matrix(kroa100_instance)
krob100_distance_matrix = calc_distance_matrix(krob100_instance)

# nn
first_cycle_kroa100, second_cycle_kroa100 = nn_alg(kroa100_distance_matrix)
first_cycle_krob100, second_cycle_krob100 = nn_alg(krob100_distance_matrix)


print(get_cycles_distance(kroa100_distance_matrix, first_cycle_kroa100, second_cycle_kroa100))
print(get_cycles_distance(krob100_distance_matrix, first_cycle_krob100, second_cycle_krob100))

plot_result(kroa100_instance, first_cycle_kroa100, second_cycle_kroa100)
plot_result(krob100_instance, first_cycle_krob100, second_cycle_krob100)

# Greedy cycle

first_cycle_kroa100, second_cycle_kroa100 = double_greedy_cycle(kroa100_distance_matrix)
first_cycle_krob100, second_cycle_krob100 = double_greedy_cycle(krob100_distance_matrix)

plot_result(kroa100_instance, first_cycle_kroa100, second_cycle_kroa100)
plot_result(krob100_instance, first_cycle_krob100, second_cycle_krob100)

print(get_cycles_distance(kroa100_distance_matrix, first_cycle_kroa100, second_cycle_kroa100))
print(get_cycles_distance(krob100_distance_matrix, first_cycle_krob100, second_cycle_krob100))

#Greedy cycle seq

first_cycle_kroa100, second_cycle_kroa100 = double_greedy_cycle_seq(kroa100_distance_matrix)
first_cycle_krob100, second_cycle_krob100 = double_greedy_cycle_seq(krob100_distance_matrix)

plot_result(kroa100_instance, first_cycle_kroa100, second_cycle_kroa100)
plot_result(krob100_instance, first_cycle_krob100, second_cycle_krob100)

print(get_cycles_distance(kroa100_distance_matrix, first_cycle_kroa100, second_cycle_kroa100))
print(get_cycles_distance(krob100_distance_matrix, first_cycle_krob100, second_cycle_krob100))

#Regret

first_cycle_kroa100, second_cycle_kroa100 = double_regret_cycle2(kroa100_distance_matrix)
first_cycle_krob100, second_cycle_krob100 = double_regret_cycle2(krob100_distance_matrix)

plot_result(kroa100_instance, first_cycle_kroa100, second_cycle_kroa100)
plot_result(krob100_instance, first_cycle_krob100, second_cycle_krob100)

print(get_cycles_distance(kroa100_distance_matrix, first_cycle_kroa100, second_cycle_kroa100))
print(get_cycles_distance(krob100_distance_matrix, first_cycle_krob100, second_cycle_krob100))
