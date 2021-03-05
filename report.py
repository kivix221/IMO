from utils import *
from nn import nn_alg
from greedy_cycle import double_regret_cycle, double_greedy_cycle_seq


def test_algorithm(matrix, instance, algorithm):
    score = []
    best_cycle = ()
    min_score = np.inf
    for n in range(100):
        fc, sc = algorithm(matrix, node=n)
        score.append(get_cycles_distance(matrix, fc, sc)[0])
        if score[-1] < min_score:
            min_score = score[-1]
            best_cycle = (fc, sc)

    plot_result(instance, *best_cycle)
    return np.mean(score), min_score, max(score)


if __name__ == '__main__':
    kroa100_instance = load_instance('kroa100.tsp')
    krob100_instance = load_instance('krob100.tsp')

    kroa100_distance_matrix = calc_distance_matrix(kroa100_instance)
    krob100_distance_matrix = calc_distance_matrix(krob100_instance)

    print("Instancja kroa100")
    print(test_algorithm(kroa100_distance_matrix, kroa100_instance, nn_alg))
    print(test_algorithm(kroa100_distance_matrix, kroa100_instance, double_greedy_cycle_seq))
    print(test_algorithm(kroa100_distance_matrix, kroa100_instance, double_regret_cycle))

    print("Instancja krob100")
    print(test_algorithm(krob100_distance_matrix, krob100_instance, nn_alg))
    print(test_algorithm(krob100_distance_matrix, krob100_instance, double_greedy_cycle_seq))
    print(test_algorithm(krob100_distance_matrix, krob100_instance, double_regret_cycle))
