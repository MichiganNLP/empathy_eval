"""
Utility functions for loading empathetic data.
"""
import os
import json
import csv
import ast
from constants import (
    EMP_HOME,
    #EMPATHETIC_DIALOGUE_DATA_DIR,
)


def load_data_file(filepath):
    """
    Load data from .txt file.
    """

    if filepath.endswith(".txt"):
        with open(filepath, "r") as file_p:
            data = file_p.readlines()
    elif filepath.endswith(".json"):
        with open(filepath, "r") as file_p:
            data = json.load(file_p)
    else:
        raise ValueError("Invalid file type %s." % filepath)

    return data


def load_emphi_data(filepath):
    """
    Load EmpHi response data.
    """
    with open(filepath, "r") as file_p:
        data = json.load(file_p)

    responses = [resp_obj["response"] for resp_obj in data]
    return responses


def load_empathetic_dialogue_data_json(
    filepath,
):
    """
    Load ED data.
    """
    with open(filepath, "r") as file_p:
        data = json.load(file_p)
    return data


def load_empathetic_dialogue_data_csv(filepath):
    """
    Load EmpatheticDialogue data.
    """
    with open(filepath, "r") as file:
        reader = csv.reader(file)
        next(reader)
        last_idx = ""
        conversation = []
        conversations = []
        for line in reader:
            conversation_idx = line[0]
            utterance = line[5]
            if last_idx == conversation_idx:
                conversation.append(utterance)
            if last_idx != conversation_idx:
                if last_idx != "":
                    conversations.append(conversation)
                    conversation = []
                conversation.append(utterance)
                last_idx = conversation_idx
    return conversations


def extract_preds(data, prefix):
    """
    Get list of prediction utterances and sentences.
    """
    preds = []
    all_sents = []
    for idx, line in enumerate(data):
        if not line.startswith(prefix):
            continue
        line = line.replace(prefix, "").strip("\n").strip()
        preds.append(line)

        sents = line.replace("! ", ". ").split(".")
        sents = [
            sent.replace("?", "").replace("!", "").strip()
            for sent in sents
            if sent != ""
        ]
        sents = [x for x in sents if x != ""]
        all_sents.extend(sents)

    print("number of predictions:", len(preds))
    print("number of sentences:", len(all_sents))
    print("number of unique predictions:", len(list(set(preds))))
    print("number of unique sentences:", len(list(set(all_sents))))

    return preds, all_sents


def extract_contexts(data, prefix):
    """
    Get list of contexts.
    """
    all_contexts = []
    for line in data:
        _context = []
        if not line.startswith(prefix):
            continue
        line = line.replace(prefix, "").strip("\n").strip()
        _context = ast.literal_eval(line)
        all_contexts.append(_context)
    return all_contexts


def save_templates(templates, hashmap, filepath):
    """
    Dump templates and hashmap to file.
    """
    with open(filepath, "w") as file_p:
        json.dump(
            {"templates": templates, "hashmap": hashmap}, file_p, indent=2
        )


def load_templates(filepath):
    """
    Load templates and hashmap from file.
    """
    with open(filepath, "r") as file_p:
        data = json.load(file_p)
    return data["templates"], data["hashmap"]


def load_empathetic_dialogue(split: str):
    """
    :split: ["train", "test", "valid"]
    """
    if split not in ["train", "valid", "test"]:
        raise ValueError("Invalid split.")
    filepath = os.path.join(EMPATHETIC_DIALOGUE_DATA_DIR, "%s.csv" % split)

    data = []
    with open(filepath, "r") as file_p:
        reader = csv.DictReader(file_p)
        for row in reader:
            data.append(
                {
                    "conv_id": row["conv_id"],
                    "utterance_idx": row["utterance_idx"],
                    "context": row["context"],
                    "prompt": row["prompt"],
                    "speaker_idx": row["speaker_idx"],
                    "utterance": row["utterance"],
                    "selfeval": row["selfeval"],
                    "tags": row["tags"],
                }
            )
    return data
