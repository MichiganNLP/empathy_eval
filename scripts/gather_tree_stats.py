"""
Script for building trees for epitome spans.
"""
from typing import List

import os
import json
from nltk.tokenize import word_tokenize
from scripts.ngrams import SpanProcessor, span_frequency
from scripts.tree import Tree, num_nodes, compression
from constants import EMP_HOME

OUTPUTS_HOME = os.path.join(EMP_HOME, "data/")


def print_stats(responses: List[str], spanifier: SpanProcessor):
    """
    Print stats.
    """
    tokenized_responses = [word_tokenize(resp) for resp in responses]
    unfolded_tree = Tree()
    unfolded_tree.build_tree(tokenized_responses, {})

    span_count = span_frequency(spanifier.templates, spanifier.hashmap)
    most_common = span_count.most_common(10)

    print("Span frequency")
    for _count_obj in most_common:
        print("  %s, %d" % (_count_obj[0], _count_obj[1]))
        print("  %s" % " ".join(spanifier.hashmap[_count_obj[0]]))
    print("Number of unique templates")
    print(len(spanifier.uniq_templates))
    print("Number of unique responses")
    print(len(list(set(responses))))

    if len(list(set(responses))) != len(spanifier.uniq_templates):
        print("Hmm... not sure why/how this can happen...")

    print("Number of nodes.")
    print(num_nodes(spanifier.tree.root))
    print("Number of span nodes.")
    print(len(span_count.keys()))
    print("Number of root children.")
    print(len(spanifier.tree.root.children))

    print("Number of unique starting words")
    print(len(unfolded_tree.root.children))
    print(unfolded_tree.root.children)

    print("Compression Ratio")
    print(num_nodes(spanifier.tree.root) / num_nodes(unfolded_tree.root))

    print("------------------------")


def _load_data(responses_filepath):
    """
    Load responses.
    """
    with open(responses_filepath, "r") as file_p:
        data = json.load(file_p)
    return [x["response"] for x in data]


def gather_stats(responses_filepath, ngrams_dir):
    """ Spanify epitome responses for dialogue systems """
    print("------------------------")
    print(" * Empathetic System: %s" % ngrams_dir)

    # dec = dec.lower()[:-1]

    responses = _load_data(responses_filepath)
    spanifier = SpanProcessor("empty_cache")
    spanifier.init_from_file(ngrams_dir)
    print_stats(responses, spanifier)


if __name__ == "__main__":
    system_name = "emocause"
    response_filepath = os.path.join(
        OUTPUTS_HOME, f"outputs/{system_name}_responses.json"
    )
    ngrams_dir = os.path.join(OUTPUTS_HOME, f"ngrams/{system_name}")
    gather_stats(response_filepath, ngrams_dir)
