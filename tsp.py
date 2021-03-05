from utils import load_instance, calc_distance_matrix, plot_result
from nn import nn_alg
from greedy_cycle import *

##########
kroa100_instance = load_instance('kroa100.tsp')
krob100_instance = load_instance('krob100.tsp')

kroa100_distance_matrix = calc_distance_matrix(kroa100_instance)
krob100_distance_matrix = calc_distance_matrix(krob100_instance)

first_cycle_kroa100, second_cycle_kroa100 = nn_alg(kroa100_distance_matrix)
first_cycle_krob100, second_cycle_krob100 = nn_alg(krob100_distance_matrix)

# print(first_cycle_kroa100)
# print(second_cycle_kroa100)

plot_result(kroa100_instance, first_cycle_kroa100, second_cycle_kroa100)
plot_result(krob100_instance, first_cycle_krob100, second_cycle_krob100)

#    Greedy cycle

first_cycle_kroa100, second_cycle_kroa100 = double_greedy_cycle(kroa100_distance_matrix)
first_cycle_krob100, second_cycle_krob100 = double_greedy_cycle(krob100_distance_matrix)

plot_result(kroa100_instance, first_cycle_kroa100, second_cycle_kroa100)
plot_result(krob100_instance, first_cycle_krob100, second_cycle_krob100)
