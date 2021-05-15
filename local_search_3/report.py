import copy
try:
    from .iterated_ls import *
except Exception:
    from local_search_3.iterated_ls import *
from tqdm import tqdm
# import ray  # pip install ray[default]

ITER = 10
MSLS_MEAN_TIME = 50


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


def run_report(algo: Algorithm.__class__, matrix, instance, params):
    algo = algo(matrix)
    params['instance'] = instance
    s, t, bc = [], [], []
    for i in tqdm(range(ITER)):
        st = time()
        (c1, c2), _ = algo.run(**params)
        st = time() - st
        s.append(get_cycles_distance(matrix, c1, c2)[0])
        t.append(st)
        bc.append((c1, c2))
    best_c = bc[np.argmin(s)]
    trip_s = calc_triple(s)
    trip_t = calc_triple(t)
    return trip_s, trip_t, best_c


def triple_to_string(trip, to_int=True):
    if to_int:
        return f"{int(trip[0])} ({int(trip[1])} - {int(trip[2])})"
    else:
        return f"%.3f (%.3f - %.3f)" % (trip[0], trip[1], trip[2])


krox = {0: 'kroa200-', 1: 'krob200-'}


def test_report_wrapper(results, names, algorithms, matrices, instances, params):
    for i, (matrix, instance) in enumerate(zip(matrices, instances)):
        for name, algo, param in zip(names, algorithms, params):
            s, t, c = run_report(algo, matrix, instance, param)
            results['score'][i].append(triple_to_string(s))
            results['time'][i].append(triple_to_string(t, False))
            plot_result(instance, c[0], c[1], krox[i] + str(algo([])))

    return results


def print_results(results, names):
    print('======SCORES======')
    print('    \t                 KROA |                  KROB')
    [print("%s\t%s | %s" % (n, s1, s2)) for n, s1, s2 in zip(names, results['score'][0], results['score'][1])]
    print('======TIMES======')
    print('    \t                 KROA |                  KROB')
    [print("%s\t%s | %s" % (n, s1, s2)) for n, s1, s2 in zip(names, results['time'][0], results['time'][1])]


if __name__ == "__main__":
    # GRID SEARCH
    # ray.init()
    # sm_perturb = SmallPerturbation()
    # lg_perturb = LargePerturbation()
    # params_sm = {'p': (sm_perturb,), 'size': np.arange(5, 21, 5), 'stop_time': (MSLS_MEAN_TIME,),
    #              'regret_begin': (True, False)}
    # params_lg = {'p': (lg_perturb,), 'size': np.arange(0.06, 0.31, 0.08), 'rand': np.arange(0.0, 1.1, 0.33),
    #              'stop_time': (MSLS_MEAN_TIME,), 'regret_begin': (True, False)}
    # params_sm = generate_params((params_sm,))
    # params_lg = generate_params((params_lg,))
    # print(params)
    # print(len(params))
    # print(next(params))

    # ka200_instance = load_instance('../data/kroa100.tsp')
    # kb200_instance = load_instance('../data/krob100.tsp')

    # ka200_dm = calc_distance_matrix(ka200_instance)
    # kb200_dm = calc_distance_matrix(kb200_instance)

    # sa = run_tests(IteratedLS, ka200_dm, ka200_instance, params_sm)
    # la = run_tests(IteratedLSa, ka200_dm, ka200_instance, params_lg)

    # print('=======SMALL========')
    # for a, p in zip(sa, params_sm):
    #     print(p)
    #     print(a)
    #     print()
    # print('=======LARGE========')
    # for a, p in zip(la, params_lg):
    #     print(p)
    #     print(a)
    #     print()

    ka200_instance = load_instance('../data/kroa200.tsp')
    kb200_instance = load_instance('../data/krob200.tsp')

    ka200_dm = calc_distance_matrix(ka200_instance)
    kb200_dm = calc_distance_matrix(kb200_instance)

    results = {'score': ([None], [None]), 'time': ([None], [None])}
    # Zamiast None trzeba wrzucić wyniki z MSLS i ustawić MSLS_MEAN_TIME na średni czas wykonywania

    str_alg = ('MSLS', 'ILS1', 'ILS2', 'ILS2a')
    sm_perturb = SmallPerturbation()
    lg_perturb = LargePerturbation()
    params_sm = {'p': sm_perturb, 'size': 10, 'stop_time': MSLS_MEAN_TIME, 'regret_begin': True}
    params_lg = {'p': lg_perturb, 'size': 0.3, 'rand': 0.6,
                 'stop_time': MSLS_MEAN_TIME, 'regret_begin': True}
    params_lga = {'p': lg_perturb, 'size': 0.3, 'rand': 0.99,
                  'stop_time': MSLS_MEAN_TIME, 'regret_begin': True}

    results = test_report_wrapper(results, str_alg[1:], [IteratedLS, IteratedLS, IteratedLSa], [ka200_dm, kb200_dm],
                                  [ka200_instance, kb200_instance], [params_sm, params_lg, params_lga])
    print_results(results, str_alg)
