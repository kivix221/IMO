try:
    from ..utils import *
except Exception:
    from utils import *
import numpy as np


def calculate_candidates(dm: np.ndarray, n: int = 10):
    candidates = np.empty((len(dm), n), dtype=np.uint16)
    tmp_min = np.empty((n,), dtype=np.uint16)

    for i, d in enumerate(dm):
        tmp_min.fill(np.argmax(d))
        for j, v in enumerate(d):
            if v < d[tmp_min[-1]]:
                if v >= d[tmp_min[-2]]:
                    tmp_min[-1] = j
                else:
                    for m, t_min in enumerate(tmp_min):
                        if v < d[t_min]:
                            tmp_min[(m+1):] = tmp_min[m:-1]
                            tmp_min[m] = j
                            break
        candidates[i] = tmp_min
    return candidates


if __name__ == "__main__":
    ka200_instance = load_instance('../data/kroa200.tsp')
    kb200_instance = load_instance('../data/krob200.tsp')

    ka200_dm = calc_distance_matrix(ka200_instance)
    kb200_dm = calc_distance_matrix(kb200_instance)

    calculate_candidates(ka200_dm)
    calculate_candidates(kb200_dm)
