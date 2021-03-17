import matplotlib.pyplot as plt
import numpy as np
import math
from random import shuffle

from typing import List


def load_instance(instance_file):
    instance = open(instance_file)
    temp_list = []

    for line in instance.readlines():
        if line[0].isdigit():
            temp_list.append(line.replace("\n", "").split(" ")[1:])

    return np.array(temp_list, dtype=int)


def calc_distance_matrix(cities):
    dist_matrix = []
    for i in range(len(cities)):
        row = []
        for j in range(len(cities)):
            if i == j:
                dist = np.inf
            else:
                dist = round(math.sqrt((cities[i, 0] - cities[j, 0]) ** 2 + (cities[i, 1] - cities[j, 1]) ** 2))
            row.append(dist)
        dist_matrix.append(row)

    return np.array(dist_matrix)


def plot_result(instance, first_cycle, second_cycle):
    first_cycle_to_plot = np.asfarray([instance[i] for i in first_cycle])
    second_cycle_to_plot = np.asfarray([instance[i] for i in second_cycle])

    plt.figure()
    plt.scatter(instance[:, 0], instance[:, 1], color='black')
    plt.plot(first_cycle_to_plot[:, 0], first_cycle_to_plot[:, 1], color='red', label = 'Cykl 1')
    plt.plot(second_cycle_to_plot[:, 0], second_cycle_to_plot[:, 1], color='blue', label = 'Cykl 2')
    plt.legend()
    plt.show()


def get_cycle_distance(matrix: np.ndarray, cycle: List[int]) -> int:
    distance = 0
    for pr, nx in zip(cycle[:-1], cycle[1:]):
        distance += matrix[pr, nx]

    return int(distance)


def get_cycles_distance(matrix: np.ndarray, cycle1: List[int], cycle2: List[int]) -> (int, int, int):
    dis1 = get_cycle_distance(matrix, cycle1)
    dis2 = get_cycle_distance(matrix, cycle2)
    return dis1+dis2, dis1, dis2


def get_random_cycle(n=100) -> (List[int], List[int]):
    whole = list(range(n))
    shuffle(whole)
    return np.array(whole[(n//2+1):]), np.array(whole[:(n//2+1)])
