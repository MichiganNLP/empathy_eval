"""
Utility functions for building ngram templates.
Mostly utility functions for doing parallelization.
"""
from typing import List, Tuple

import sys
import time
from collections import defaultdict, Counter
from pathos.multiprocessing import ProcessingPool as Pool
from nltk import FreqDist
from nltk.util import ngrams


def merge_sum(dicts):
    """
    Merge dictionaries by summing
    """
    #ret = defaultdict(int)
    #for _d in dicts:
    #    for k, v in _d.items():
    #        ret[k] += v
    #return dict(ret)
    if len(dicts) < 1:
        raise RuntimeError("Expected a non-empty list of count dictionaries.")

    counts = Counter(dicts[0])
    for _dict in dicts[1:]:
        counts = counts + _dict
    return dict(counts)


def merge_append(dicts):
    """
    Merge dictionaries by appending
    """
    ret = defaultdict(list)
    for _d in dicts:
        for k, v in _d.items():
            ret[k].extend(v)
    return dict(ret)


def run_parallel(data: List[List[str]], pool_size: int):

    ngram_idxs = defaultdict(list)
    max_len = max([len(x) for x in data])
    chunk_groups = list(enumerate(data))
    chunk_size = len(chunk_groups) // pool_size
    counter = {}

    def count_func(_data: List[Tuple[int, List[str]]]):
        """
        Count ngrams.
        """
        _counter = FreqDist()
        _ngram_idxs = defaultdict(list)
        for idx, (group_idx, tokens) in enumerate(_data):
            for n in range(2, max_len + 1):
                _ngrams = list(ngrams(tokens, n))
                for ngram in _ngrams:
                    # orig_idx = chunk_size * group_idx + idx
                    _ngram_idxs[ngram].append(group_idx)
                _counter.update(_ngrams)
        return _counter, _ngram_idxs

    if chunk_size < 1:
        breakpoint()

    chunks = [
        chunk_groups[i : i + chunk_size]
        for i in range(0, len(chunk_groups), chunk_size)
    ]

    pool = Pool(pool_size)
    #pool.restart()
    try:
        _results = pool.amap(count_func, chunks)
        while not _results.ready():
            time.sleep(1)
        results = _results.get()
        count_results = [x[0] for x in results]
        _ngram_idxs = [x[1] for x in results]

        counter = merge_sum([counter] + count_results)
        ngram_idxs = merge_append([ngram_idxs] + _ngram_idxs)
    except (KeyboardInterrupt, SystemExit):
        pool.terminate()
        sys.exit(1)

    return ngram_idxs, counter
