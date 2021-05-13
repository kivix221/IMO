import copy

import matplotlib.pyplot as plt
import numpy as np

try:
    from .iterated_ls import *
except Exception:
    from local_search_3.iterated_ls import *
from tqdm import tqdm
# import multiprocessing as mp
import ray

ITER = 10
MSLS_MEAN_TIME = 25


def calc_triple(tab):
    return np.mean(tab), np.min(tab), np.max(tab)


@ray.remote
def _test_alg_par(_, algo: Algorithm, kwargs):
    st = time()
    (c1, c2), a = algo.run(**kwargs)
    st = time() - st
    # cycles[_][0] = c1
    # cycles[_][1] = c2
    # times[_] = st
    # scores[_] = get_cycles_distance(algo.matrix, c1, c2)[0]
    # adds[_][0] = a[0]
    # adds[_][1] = a[1]
    return (c1, c2), st, get_cycles_distance(algo.matrix, c1, c2)[0], a


# pool = mp.Pool(min(mp.cpu_count(), ITER))
def test_alg(algo: Algorithm, **kwargs):
    assert 'instance' in kwargs.keys()
    scores = np.zeros((ITER,), dtype=np.float32)
    times = np.zeros((ITER,), dtype=np.float32)
    adds = np.zeros((ITER, 2), dtype=np.float32)
    cycles = np.zeros((ITER, 2, 51), dtype=np.float32)

    r = [_test_alg_par.remote(_, algo, kwargs) for _ in range(ITER)]
    r = ray.get(r)

    for i, ((c1, c2), st, sd, a) in enumerate(r):
        cycles[i][0] = c1
        cycles[i][1] = c2
        times[i] = st
        scores[i] = sd
        adds[i][0] = a[0]
        adds[i][1] = a[1]

    ap1 = np.average(adds[:, 0], axis=-1, weights=adds[:, 1])
    ap2 = np.sum(adds[:, 1], axis=-1)
    best_c = cycles[np.argmin(scores)]
    scores = calc_triple(scores)
    times = calc_triple(times)
    # plot_result(kwargs['instance'], best_c[0], best_c[1], title)
    # plt.imsave()
    return scores, times, best_c, (ap1, ap2)


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
    s, t, bc, a = [], [], [], []
    for p in tqdm(params):
        ts, tt, tbc, _ = test_alg(algo, instance=instance, **p)
        s.append(ts)
        t.append(tt)
        bc.append(tbc)
        a.append(_)
    bsi = np.argmin(np.array(s)[:, 0])
    print(params[bsi])
    print(s[bsi])
    print(t[bsi])
    # plot_result(instance, bc[bsi][0], bc[bsi][1], s[bsi][0])
    # plt.savefig('test.jpg')
    return a


if __name__ == "__main__":
    ray.init()
    sm_perturb = SmallPerturbation()
    lg_perturb = LargePerturbation()
    params_sm = {'p': (sm_perturb,), 'size': np.arange(5, 21, 5), 'stop_time': (MSLS_MEAN_TIME,)}
    params_lg = {'p': (lg_perturb,), 'size': np.arange(0.06, 0.31, 0.08), 'rand': np.arange(0.0, 1.1, 0.33),
                 'stop_time': (MSLS_MEAN_TIME,), 'regret_begin': (True, False)}
    params_sm = generate_params((params_sm,))
    params_lg = generate_params((params_lg,))
    # print(params)
    # print(len(params))
    # print(next(params))

    ka200_instance = load_instance('IMO/data/kroa100.tsp')
    kb200_instance = load_instance('IMO/data/krob100.tsp')

    ka200_dm = calc_distance_matrix(ka200_instance)
    kb200_dm = calc_distance_matrix(kb200_instance)

    # sa = run_tests(IteratedLS, ka200_dm, ka200_instance, params_sm)
    la = run_tests(IteratedLSa, ka200_dm, ka200_instance, params_lg)

    # print('=======SMALL========')
    # for a, p in zip(sa, params_sm):
    #     print(p)
    #     print(a)
    #     print()
    print('=======LARGE========')
    for a, p in zip(la, params_lg):
        print(p)
        print(a)
        print()
