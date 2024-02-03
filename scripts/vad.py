"""
Intensity, Valence, Arousal.
"""

import os
import json
import numpy as np
from nltk.tokenize import word_tokenize
from constants import EMP_HOME

DATA_HOME = os.path.join(EMP_HOME, "data")


def load_vad():
    """
    Load vad info.
    """
    with open(os.path.join(DATA_HOME, "vad/VAD.json"), "r") as file_p:
        vad = json.load(file_p)
    return vad


def load_intensity():
    """
    Load intensity info.
    """
    with open(os.path.join(DATA_HOME, "intensity.txt"), "r") as file_p:
        intensity_lex = file_p.readlines()

    intensities = {}
    for intensity in intensity_lex:
        tokens = intensity.split()
        intensities[tokens[0]] = float(tokens[2])
    return intensities


VAD = load_vad()
INTENSITIES = load_intensity()


def get_token_vad(token):
    """
    Get VAD vector
    """
    return VAD.get(token, [0, 0, 0])


def get_token_intensity(token):
    return INTENSITIES.get(token, 0)


def get_vad(query):
    """
    Get mean, max scores for VAD.
    """
    tokens = word_tokenize(query.lower())
    vads = [get_token_vad(token) for token in tokens]
    vads = [x for x in vads if x is not None]

    valence = [x[0] for x in vads]
    arousal = [x[1] for x in vads]
    dominance = [x[2] for x in vads]
    return valence, arousal, dominance


def get_intensity(query):
    tokens = word_tokenize(query.lower())
    return [get_token_intensity(token) for token in tokens]


def get_vad_stats(data, system):
    """
    Compute intensity, vad.
    """

    results = []

    for convo_obj in data:

        context = convo_obj["query"]
        last_utt = context[-1]
        response = convo_obj["response"]

        context_v, context_a, context_d = get_vad(last_utt)
        response_v, response_a, response_d = get_vad(response)

        context_intensity = get_intensity(last_utt)
        response_intensity = get_intensity(response)

        max_v_context = 0
        max_a_context = 0
        max_d_context = 0
        mean_v_context = 0
        mean_a_context = 0
        mean_d_context = 0

        if len(context_v) > 0:
            max_v_context = max(context_v)
            mean_v_context = np.mean(context_v)
        if len(context_a) > 0:
            max_a_context = max(context_a)
            mean_a_context = np.mean(context_a)
        if len(context_d) > 0:
            max_d_context = max(context_d)
            mean_d_context = np.mean(context_d)

        if len(response_v) > 0:
            max_v = max(response_v)
            mean_v = np.mean(response_v)
        if len(response_a) > 0:
            max_a = max(response_a)
            mean_a = np.mean(response_a)
        if len(response_d) > 0:
            max_d = max(response_d)
            mean_d = np.mean(response_d)

        diff_max_v = max_v_context - max_v
        diff_mean_v = mean_v_context - mean_v
        diff_max_a = max_a_context - max_a
        diff_mean_a = mean_a_context - mean_a
        diff_max_d = max_d_context - max_d
        diff_mean_d = mean_d_context - mean_d
        diff_intensity = max(context_intensity) - max(response_intensity)

        results.append(
            {
                "max_v": max_v,
                "mean_v": mean_v,
                "max_a": max_a,
                "mean_a": mean_a,
                "max_d": max_d,
                "mean_d": mean_d,
                "diff_max_v": diff_max_v,
                "diff_mean_v": diff_mean_v,
                "diff_max_a": diff_max_a,
                "diff_mean_a": diff_mean_a,
                "diff_max_d": diff_max_d,
                "diff_mean_d": diff_mean_d,
                "diff_max_intensity": diff_intensity,
            }
        )

    return results


def compare_vad(filepaths):
    """ Compare VADs """
    scores = {}
    for system, filepath in filepaths:
        with open(filepath, "r") as file_p:
            data = json.load(file_p)

        vad_stats = get_vad_stats(data, system)

        diff_max_v = np.mean([x["diff_max_v"] for x in vad_stats])
        diff_max_a = np.mean([x["diff_max_a"] for x in vad_stats])
        diff_max_d = np.mean([x["diff_max_d"] for x in vad_stats])
        diff_max_intensity = np.mean(
            [x["diff_max_intensity"] for x in vad_stats]
        )

        print("--")
        print("--")
        print("--")
        print(f"({system}) Diff Max V: {diff_max_v}")
        print(f"({system}) Diff Max A: {diff_max_a}")
        print(f"({system}) Diff Max D: {diff_max_d}")
        print(f"({system}) Diff Intensity: {diff_max_intensity}")

        scores[system] = {
            "diff_max_v": diff_max_v,
            "diff_max_a": diff_max_a,
            "diff_max_d": diff_max_d,
            "diff_max_intensity": diff_max_intensity,
        }
    return scores


def main():
    """ Driver """
    systems = [
        "trs",
        "moel",
        "mime",
        "emocause",
        "cem",
        "kemp",
        "seek",
        "care",
        "emphi_2",
        "human",
    ]
    outputs_dir = os.path.join(EMP_HOME, "data/outputs/")
    filepaths = [
        (system, os.path.join(outputs_dir, f"{system}_responses.json"))
        for system in systems
    ]

    compare_vad(filepaths)


if __name__ == "__main__":
    main()
