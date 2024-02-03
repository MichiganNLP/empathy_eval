"""
SpanProcessor Module
SpanProcessor is responsible for "spanifying" responses,
folding/unfolding operations, and building response trees.
"""
from typing import List, Dict, Tuple

import sys
import os
import time
import argparse
import pickle
import json
from collections import defaultdict, Counter
from nltk import FreqDist
from nltk.tokenize import word_tokenize
from nltk.util import ngrams

from pathos.multiprocessing import ProcessingPool as Pool

from scripts.data_utils import (
    load_data_file,
)
from scripts.ngram_utils import run_parallel
from scripts.tree import Tree
from constants import EMP_HOME


PYTHONHASHSEED = 0


def get_all_ngrams(tokens: List[str]):
    """
    Get all ngrams of tokenized query.
    """
    _ngrams = []
    max_len = len(tokens)
    for n in range(2, max_len + 1):
        _ngrams.extend(ngrams(tokens, n))
    return _ngrams


class SpanProcessor:
    """
    SpanProcessor:
    Responsible for "spanifying" dialogue responses,
    supporting folding and unfolding operations,
    as well as building and maintaining response trees.
    """

    def __init__(self, cache_dir, default_output_dir=None):
        self.hashmap = None
        self.templates = None
        self.leaf_hashes = None
        self.parent_hashes = None
        self.counts = None
        self.ngram_idxs = None
        self.tree = Tree()
        self.parents = None
        self.weights = None
        self.span_hashes = None
        self.default_output_dir = default_output_dir
        self.cache_dir = cache_dir
        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)
        self.cache_every = 1

    @staticmethod
    def hash_ngram(ngram):
        """ Hash ngram """
        return "span_%s" % str(hash("__".join(ngram)))[1:17]

    def hashmap_size(self):
        """ Number of hashes """
        return len(self.hashmap)

    @staticmethod
    def compute_weight(ngram, _count):
        """
        Heuristic for scoring spans.
        It would be nice to make this an abstract method
        that is easily extendable.
        """
        if _count == 1:
            return _count
        return len(ngram) * _count

    def cache(self, data):
        with open(
            os.path.join(self.cache_dir, "__dict__.pkl"), "wb"
        ) as file_p:
            pickle.dump(self.__dict__, file_p)

        with open(
            os.path.join(self.cache_dir, "__data__.pkl"), "wb"
        ) as file_p:
            pickle.dump(data, file_p)

    def load_from_cache(self, cache_dir):
        with open(os.path.join(cache_dir, "__dict__.pkl"), "rb") as file_p:
            tmp_dict = pickle.load(file_p)
        # self.__dict__.clear()
        for k, v in tmp_dict.items():
            self.__dict__[k] = v

        with open(os.path.join(cache_dir, "__data__.pkl"), "rb") as file_p:
            data = pickle.load(file_p)
        return self, data

    def resume_spanify(self, data, pool_size=1):
        """
        Resume spanifying from cache
        """
        hashseed = os.getenv("PYTHONHASHSEED")
        if not hashseed:
            raise RuntimeError("Python hash seed not set!")
        done = False
        while not done:
            print(f"Counter: {self.counter}")
            start = time.time()
            data, done = self.spanify_single(data, pool_size=pool_size)
            end = time.time()
            print(f"Iteration took {end - start}")
            self.counter += 1
            if self.counter % self.cache_every == 0:
                self.cache(data)

        self.templates = data
        self.leaf_hashes = {
            _hash: tokens
            for _hash, tokens in self.hashmap.items()
            if not any([token.startswith("span_") for token in tokens])
        }
        self.parent_hashes = {
            _hash: tokens
            for _hash, tokens in self.hashmap.items()
            if _hash not in self.leaf_hashes
        }
        self.span_hashes = {
            span: _hash for _hash, span in self.hashmap.items()
        }
        return self.templates

    def spanify(self, data: List[List[str]], pool_size=1) -> List[List[str]]:
        """
        Convert text data into a collection of spans.
        :data: List of tokenized responses.
        Returns templates, aka the data after spans are replaced with hashes.
        """
        hashseed = os.getenv("PYTHONHASHSEED")
        if not hashseed:
            raise RuntimeError("Python hash seed not set!")
        self.hashmap = {}
        self.counts = FreqDist()
        self.ngram_idxs = defaultdict(list)
        self.parents = defaultdict(list)
        done = False
        self.counter = 0
        while not done:

            print(f"Counter: {self.counter}", flush=True)
            start = time.time()
            data, done = self.spanify_single(data, pool_size=pool_size)
            end = time.time()
            print(f"Iteration took {end - start}", flush=True)
            self.counter += 1
            if self.counter % self.cache_every == 0:
                self.cache(data)

        self.templates = data
        self.leaf_hashes = {
            _hash: tokens
            for _hash, tokens in self.hashmap.items()
            if not any([token.startswith("span_") for token in tokens])
        }
        self.parent_hashes = {
            _hash: tokens
            for _hash, tokens in self.hashmap.items()
            if _hash not in self.leaf_hashes
        }
        self.span_hashes = {
            span: _hash for _hash, span in self.hashmap.items()
        }
        return self.templates

    def build_tree(self):
        """
        Build tree from templates and hashes.
        """
        if self.templates is None:
            raise RuntimeError("self.templates is not set!")
        if self.hashmap is None:
            raise RuntimeError("self.hashmap is not set!")

        self.tree.build_tree(self.templates, self.hashmap)

    @staticmethod
    def count_func(
        data: List[List[str]],
    ) -> Tuple[Dict[str, List[int]], Dict[str, int]]:
        """
        Count ngrams in data, where data is a list of tokenized responses.

        Returns
        :ngram_idxs: Mapping of ngram to list of response indices in which
            the ngram appears.
        :counts: Counter for each ngram.
        """
        ngram_idxs = defaultdict(list)
        counts = FreqDist()
        if len([len(x) for x in data]) < 1:
            raise RuntimeError("Count_func broke?")
        max_len = max([len(x) for x in data])
        for idx, tokens in enumerate(data):
            for n in range(2, max_len + 1):
                _ngrams = list(ngrams(tokens, n))
                for ngram in _ngrams:
                    if idx not in ngram_idxs[ngram]:
                        ngram_idxs[ngram].append(idx)
                counts.update(_ngrams)
        return ngram_idxs, counts

    def spanify_single(
        self, data: List[List[str]], pool_size=1
    ) -> Tuple[List[List[str]], bool]:
        """
        'Spanify' data by converting common patterns into spans,
        where data is a list of tokenized responses.

        Returns the data, where common ngrams are replaced by the pans,
        as well as a boolean flag indicating whether we are done spanifying.
        """
        if pool_size == 1:
            ngram_idxs, counter = self.count_func(data)

        else:
            print("Starting parallel compute...", flush=True)
            start = time.time()
            ngram_idxs, counter = run_parallel(data, pool_size)
            print(
                f"Parallel compute finished, took {time.time() - start}",
                flush=True,
            )

        start = time.time()
        max_count = max(counter.values())
        if max_count < 2:
            return data, True

        span_weights = sorted(
            [
                (ngram, self.compute_weight(ngram, count))
                for ngram, count in counter.items()
            ],
            key=lambda x: x[1],
        )

        if self.weights is None:
            self.weights = {ngram: weight for (ngram, weight) in span_weights}
        else:
            for ngram, weight in span_weights:
                if ngram in self.weights:
                    continue
                self.weights[ngram] = weight

        max_weight = span_weights[-1][1]

        # Tie breaker: TODO: Make this customizeable somehow.
        potential_spans = [
            _span for _span in span_weights if _span[1] == max_weight
        ]
        span_tuple = min(potential_spans, key=lambda x: len(x[0]))
        span = span_tuple[0]
        orig_posts_idxs = ngram_idxs[span]
        if len(orig_posts_idxs) < 1:
            raise RuntimeError("Should never reach this point?")

        _hash = self.hash_ngram(span)
        if _hash in self.hashmap:
            raise RuntimeError("Hash collision!")
        self.hashmap[_hash] = span

        span_str = " ".join(span)
        for _idx in orig_posts_idxs:
            orig_data = " ".join(data[_idx])
            hashed = orig_data.replace(span_str, _hash)
            data[_idx] = word_tokenize(hashed)
        print(f"Rest took {time.time() - start}")

        return data, False

    def get_parents(self, _hash):
        """
        Get path from a tree node to root.
        """
        all_parents = []
        visited = set()
        queue = [_hash]
        while len(queue) > 0:
            curr = queue[0]
            queue = queue[1:]

            if curr not in self.parents:
                continue
            if curr in visited:
                continue
            visited.add(curr)

            all_parents.append(curr)
            parents = self.parents[curr]
            queue = queue + parents

        return all_parents

    def resolve(self, span: Tuple[str], parent_hash: str) -> str:
        """
        Resolve a span to its original text.
        Requires recursively resolving inner-spans if any.
        """
        if self.parents is None:
            raise RuntimeError("self.parents is not set!")

        resolved = []
        for token in span:
            if token in self.hashmap:
                self.parents[token].append(parent_hash)
                resolved.append(self.resolve(self.hashmap[token], token))
            else:
                resolved.append(token)
        return " ".join(resolved)

    def unfold_hashmap(self) -> Dict[str, str]:
        """
        Unfold all hashes in self.hashmap.
        Return a dictionary of hashes to unfoled response utterances.
        """
        if self.hashmap is None:
            raise RuntimeError("self.hashmap is not set!")

        unfolded = {}
        for _hash, span in self.hashmap.items():
            unfolded[_hash] = self.resolve(span, _hash)
        return unfolded

    def encode(self, query: str):
        """
        Encode a query based on self.weights
        """
        if self.weights is None:
            raise RuntimeError("self.weights is not constructed.")

        done = False
        while not done:
            query, done = self._encode_single(query)
            print(query)
        return query

    def _encode_single(self, query: str):
        """
        Inner encode function.
        Returns the encoded query, and a boolean flag
        to indicate whether encoding is done.
        """
        tokens = word_tokenize(query)
        _ngrams = get_all_ngrams(tokens)

        weights = [
            (_ngram, self.weights.get(_ngram, 0))
            for _ngram in _ngrams
            if _ngram in self.span_hashes
        ]
        if len(weights) < 1:
            return " ".join(tokens), True

        max_weight = max(weights, key=lambda x: x[1])
        if max_weight[1] == 1:
            return " ".join(tokens), True

        _span = max_weight[0]
        print(f"span: {_span}")
        _hash = self.hash_ngram(_span)
        encoded = " ".join(tokens).replace(" ".join(_span), _hash)
        return encoded, False

    def dump(self, output_dir=None):
        """
        Dump self to file.
        """
        if output_dir is None:
            if self.default_output_dir is None:
                raise ValueError("Missing output dir.")

            output_dir = self.default_output_dir

        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        weights_file = os.path.join(output_dir, "weights.pickle")
        with open(weights_file, "wb") as file_p:
            pickle.dump(self.weights, file_p)

        hashmap_file = os.path.join(output_dir, "hashmap.json")
        with open(hashmap_file, "w") as file_p:
            json.dump(self.hashmap, file_p, indent=4)

        templates_file = os.path.join(output_dir, "templates.json")
        with open(templates_file, "w") as file_p:
            json.dump(self.templates, file_p, indent=4)

    def init_from_file(self, input_dir):
        """
        Load from file.
        """
        weights_file = os.path.join(input_dir, "weights.pickle")
        with open(weights_file, "rb") as file_p:
            self.weights = pickle.load(file_p)

        hashmap_file = os.path.join(input_dir, "hashmap.json")
        with open(hashmap_file, "r") as file_p:
            self.hashmap = json.load(file_p)

        templates_file = os.path.join(input_dir, "templates.json")
        with open(templates_file, "r") as file_p:
            self.templates = json.load(file_p)
        self.uniq_templates = list(
            set([tuple(_templ) for _templ in self.templates])
        )

        self.span_hashes = {
            tuple(_span): _hash for _hash, _span in self.hashmap.items()
        }
        self.build_tree()


def tag_tokenize(data: str) -> List[str]:
    """
    Tag tokens with '_%d', where d = count in which token appears in sentence.
    Ex: "I am who I am" --> ['I_1', 'am_1', 'who_1', 'I_2', 'am_2']
    """
    data = [word_tokenize(sent) for sent in data]
    ret = []

    for sent in data:
        tagged = []
        for idx, token in enumerate(sent):
            tagged.append("%s_%d" % (token, sent[: idx + 1].count(token)))
        ret.append(tagged)
    return ret


def _span_frequency_count(tokens, hashmap, span_freq):
    """
    Recursive helper function.
    """
    for token in tokens:
        span_freq[token] += 1
        if token not in hashmap:
            continue
        span_ref = hashmap[token]
        _span_frequency_count(span_ref, hashmap, span_freq)


def span_frequency(data, hashmap):
    """
    Recursively count number of occurances of each template span.
    """
    span_freq = defaultdict(int)
    for sentence in data:
        _span_frequency_count(sentence, hashmap, span_freq)
    span_freq = Counter(
        {
            _hash: _count
            for _hash, _count in span_freq.items()
            if _hash.startswith("span_")
        }
    )
    return span_freq


def main():
    """
    CLI for building ngrams and spanifying data.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-source",
        "-ds",
        type=str,
        help="source of data",
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="input filename",
    )
    parser.add_argument(
        "--decoder",
        "-d",
        type=str,
        default="response",
        help="decoder to spanify",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        required=True,
        help="output directory -- files will be saved in %s with a .json extension."
        % os.path.join(EMP_HOME, "outputs/ngrams"),
    )
    parser.add_argument(
        "--from-cache",
        default=None,
        required=False,
        help="filepath to cache dir if to resume from cache.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        required=False,
        help="directory to cache to.",
    )
    args = parser.parse_args()

    data_source = args.data_source
    input_file = args.input
    decoder = args.decoder
    output_dir = args.output
    from_cache = args.from_cache
    write_cache_dir = args.cache_dir

    if from_cache is not None:
        if not os.path.isdir(from_cache):
            raise RuntimeError(f"Missing directory {from_cache}.")

        x = SpanProcessor(from_cache)
        x, data = x.load_from_cache(from_cache)
        x.resume_spanify(data, pool_size=6)
        x.dump(output_dir)
        sys.exit(0)

    if data_source is None and input_file is None:
        raise RuntimeError("Either data_source or input_file must be set!")

    if args.output is None:
        args.output = data_source

    print("Running on %s" % input_file)
    print("Running with decoder %s" % decoder)

    data = load_data_file(input_file)

    if decoder not in data[0]:
        raise RuntimeError("Invalid decoder for %s." % input_file)

    data = [response_obj[args.decoder] for response_obj in data]


    data = [word_tokenize(sent) for sent in data]
    x = SpanProcessor(write_cache_dir)
    start = time.time()
    _ = x.spanify(data, pool_size=4)
    end = time.time()
    print("Time took ", end - start)

    x.dump(output_dir)


if __name__ == "__main__":
    main()
