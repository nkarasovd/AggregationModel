from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
from tqdm import tqdm


def bootstrap_aggregate_metric(metric: Callable[[List[float]], float], y: List[float],
                               err: float = 0.05, iters: int = 100, size: float = 1.0):
    values = []
    y = np.array(y)
    real_value = metric(y)
    n = len(y)
    sn = int(size * n)
    left = int(iters * err / 2)
    while len(values) < iters:
        inds = np.random.choice(n, sn)
        try:
            value = metric(y[inds])
            values.append(value)
        except:
            pass
    values = sorted(values)
    # return round(real_value, 4), round(values[left], 4), round(values[iters - 1 - left], 4)
    return real_value, values[left], values[iters - 1 - left]


def paper_metrics_iter(preds: Iterable[Tuple[int, Dict[int, float]]]) -> Dict[str, Tuple[float, float, float]]:
    aps = []
    correct_top = []
    for true_is_id, is_scores in tqdm(preds, position=0, leave=True):
        sorted_scores = sorted(is_scores.items(), key=lambda x: x[1], reverse=True)
        pos = float("inf")
        for i, (is_id, score) in enumerate(sorted_scores):
            if is_id == true_is_id:
                pos = i
                break
        aps.append(1 / (pos + 1))
        correct_top.append(pos)

    rrs = {f"rr@{k}": bootstrap_aggregate_metric(np.mean, [x < k for x in correct_top]) for k in [1, 5, 10]}
    scores = {"mrr": bootstrap_aggregate_metric(np.mean, aps), **rrs}

    for name, score in scores.items():
        print(f"{name}: {score}")
    return scores
