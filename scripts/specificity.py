"""
Script to compute NIDF scores.
"""

import os
import json
import pickle
import math
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from constants import EMP_HOME


def compute_nidf(responses, ed_counts):
    """
    Count up tokens.
    """
    word_counts = ed_counts["word_counts"]

    min_c = min(word_counts.values())
    max_c = max(word_counts.values())

    word2nidf = {
        word: (math.log(max_c) - math.log(_count))
        / (math.log(max_c) - math.log(min_c))
        for word, _count in word_counts.items()
    }

    nidf_scores = []
    for resp in responses:
        tokens = list(set(word_tokenize(resp)))

        nidfs = [word2nidf.get(tok, 1) for tok in tokens]
        nidf_scores.append(np.mean(nidfs))
    return nidf_scores


def count_ed(ed_files):
    """
    Count vocab in ED.
    """
    counts = Counter()
    num_sents = defaultdict(lambda: 0)
    for filepath in ed_files:
        print(f"Counting {filepath}")
        with open(filepath, "r") as file_p:
            data = file_p.readlines()
        for sample in tqdm(data[1:]):

            parts = sample.strip().split(",")
            utterance = parts[5].replace("_comma_", ",")

            tokens = list(set(word_tokenize(utterance)))
            counts.update(tokens)
            for tok in tokens:
                num_sents[tok] += 1

    return {"word_counts": counts, "num_sents": dict(num_sents)}


def main():
    """ Driver """
    outputs_dir = os.path.join(EMP_HOME, "data/outputs")

    ed_dir = os.path.join(EMP_HOME, "data/empatheticdialogues")
    ed_files = [
        os.path.join(ed_dir, "train.csv"),
        os.path.join(ed_dir, "valid.csv"),
    ]
    ed_counts_filepath = os.path.join(EMP_HOME, "data/ed_counts.pkl")
    if os.path.isfile(ed_counts_filepath):
        with open(ed_counts_filepath, "rb") as file_p:
            ed_counts = pickle.load(file_p)
    else:
        ed_counts = count_ed(ed_files)
        with open(ed_counts_filepath, "wb") as file_p:
            pickle.dump(ed_counts, file_p)

    systems = [
        "trs",
        "care",
        "cem",
        "emocause",
        "emphi",
        "human",
        "kemp",
        "mime",
        "moel",
        "seek",
    ]

    for system in systems:
        responses_file = os.path.join(outputs_dir, f"{system}_responses.json")
        with open(responses_file, "r") as file_p:
            data = json.load(file_p)

        responses = [resp_obj["response"] for resp_obj in data]
        nidfs = compute_nidf(responses, ed_counts)
        print(f"NIDF for {system}: {np.mean(nidfs)}")


if __name__ == "__main__":
    main()
