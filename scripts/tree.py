"""
Tree data structure to represent a
collection of dialogue responses
"""
import os
import json
from nltk.tokenize import word_tokenize
from constants import EMP_HOME


PYTHONHASHSEED = 1


def hash_node(token):
    """
    Get hash of node.
    """
    return str(hash(token))[1:17]


class Node:
    """ Node """

    def __init__(self, token, parent=None, is_root=False):
        self.id = hash_node(token)
        self.token = token
        self.is_root = is_root
        self.count = 0
        self.parent = parent

        self.children = {}  # Map of token to node objects

    def print_node(self, level=0):
        """ Print node and its children recursively. """

        print_string = (
            " " * level
            + "(%s: %d)" % (self.token, self.count)
            + " [%d]\n" % level
        )
        for child in self.children.values():
            child_string = child.print_node(level + 1)
            print_string += child_string
        return print_string

    def __repr__(self):
        return "(%s: %d)" % (self.token, self.count)


class Tree:
    """ Tree representation of a group of (response) utterances """

    def __init__(self):
        root_token = "__ROOT__"
        self.root = Node(root_token, is_root=True)
        self.max_node = self.root
        self.counter = {}
        self.node_pointers = {}

    def insert(self, template):
        """
        Insert token node into tree.
        """
        curr = self.root
        for token in template:

            node = curr.children.get(token)
            if node is None:
                node = Node(token, parent=curr)
                if token.startswith("span_"):
                    self.node_pointers[token] = node
                curr.children[token] = node

            node.count += 1
            self.counter[node.id] = node.count
            curr = node

    def build_tree(self, templates, hashmap):
        """
        Construct Tree from templates.
        """
        for template in templates:
            self.insert(template)


def num_nodes(node):
    """ Get number of nodes """
    if len(node.children) < 1:
        return 1

    count = 0
    queue = [node]
    while len(queue) > 0:
        curr = queue.pop()
        count += 1
        children = list(curr.children.values())
        for child_node in children:
            queue.append(child_node)
    return count


def compression(folded_tree, unfolded_tree):
    """
    Get compression rate.
    """
    return num_nodes(folded_tree.root) / num_nodes(unfolded_tree.root)


def main():
    """
    Testing out Tree.
    """
    for filename in [
        "kemp_decoder_greedy.json",
        "mime_decoder_greedy.json",
        "mime_beam_search.json",
        "mime_top_k.json",
        "moel_decoder_greedy.json",
        "moel_beam_search.json",
        "moel_top_k.json",
    ]:
        filepath = os.path.join(
            EMP_HOME, "outputs/epitome/ngrams/%s" % filename
        )
        with open(filepath, "r") as file_p:
            data = json.load(file_p)
        templates = data["templates"]
        hashmap = data["hashmap"]

        tree = Tree()
        tree.build_tree(templates, hashmap)
        print(tree.root.print_node())
        # tree.print_tree()
        print(num_nodes(tree.root))
        breakpoint()


if __name__ == "__main__":
    main()
