import copy

import matplotlib.pyplot as plt
import numpy as np
from .iterated_ls import *

ITER = 10
MSLS_MEAN_TIME = 15


def calc_triple(tab):
    return np.mean(tab), np.min(tab), np.max(tab)


def test_alg(algo: Algorithm, **kwargs):
    assert 'instance' in kwargs.keys()
    scores = []
    times = []
    adds = []
    cycles = []
    for _ in range(ITER):
        st = time()
        (c1, c2), a = algo.run(**kwargs)
        st = time() - st
        cycles.append((c1, c2))
        times.append(st)
        scores.append(get_cycles_distance(algo.matrix, c1, c2)[0])
        adds.append(a)
    best_c = cycles[np.argmin(scores)]
    scores = calc_triple(scores)
    times = calc_triple(times)
    # plot_result(kwargs['instance'], best_c[0], best_c[1], title)
    # plt.imsave()
    return scores, times, best_c, adds


def generate_params(tab_param):
    params = []
    for tp in tab_param:
        keys = list(tp.keys())
        for p in _get_all_perm(tp, 0, keys):
            params.append(p)
    return params


def _get_all_perm(one_param, curr_param, keys, ret=None):
    if curr_param is None:
        yield ret
        return
    if ret is None:
        ret = dict()

    # print(curr_param, keys)
    key = keys[curr_param]
    # print(key)
    if curr_param + 1 >= len(keys):
        n_key = None
    else:
        n_key = curr_param + 1

    for p in one_param[key]:
        r = copy.copy(ret)
        r[key] = p
        # print(r)
        yield from _get_all_perm(one_param, n_key, keys, r)


def run_tests(algo: Algorithm.__class__, matrix, instance, params):
    algo = algo(matrix)
    s, t, bc = [], [], []
    for p in params:
        ts, tt, tbc, _ = test_alg(algo, instance=instance, **p)
    bsi = np.argmin(np.array(s)[:, 0])
    plot_result(instance, bc[bsi][0], bc[bsi][1], s[bsi][0])
    plt.imsave('test.jpg')


if __name__ == "__main__":
    sm_perturb = SmallPerturbation()
    lg_perturb = LargePerturbation()
    params_sm = {'p': (sm_perturb,), 'size': np.arange(5, 21, 5), 'stop_time': (MSLS_MEAN_TIME,)}
    params_lg = {'p': (lg_perturb,), 'size': np.arange(0.05, 0.31, 0.05), 'rand': np.arange(0.0, 1.1, 0.2),
                 'stop_time': (MSLS_MEAN_TIME,)}
    params = generate_params((params_lg, params_sm))
    print(params)
    print(len(params))
    # print(next(params))

    ka200_instance = load_instance('IMO/data/kroa200.tsp')
    kb200_instance = load_instance('IMO/data/krob200.tsp')

    ka200_dm = calc_distance_matrix(ka200_instance)
    kb200_dm = calc_distance_matrix(kb200_instance)

    run_tests(IteratedLS, ka200_dm, ka200_instance, params)
