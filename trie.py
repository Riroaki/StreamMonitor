import numpy as np


class Node(object):
    """Class of a trie node."""

    def __init__(self, word: any, count: int):
        self.word = word
        self.count = count
        self.child = {}

    def is_leaf(self):
        return len(self.child) == 0


class Trie(object):
    """Class of a trie."""

    def __init__(self):
        """Init a root for the tries tree."""
        self.root = Node(-1, 0)

    def insert(self, seq: list or tuple, count: int = 1) -> None:
        """Insert a sequence into the trie."""
        curr_node = self.root
        curr_node.count += count
        for word in seq:
            if word not in curr_node.child:
                curr_node.child[word] = Node(word, count)
            curr_node = curr_node.child[word]

    def search(self, seq: list) -> bool:
        """Search a sequence path."""
        curr_node = self.root
        found = True
        for word in seq:
            if word not in curr_node.child:
                found = False
                break
            curr_node = curr_node.child[word]
        return found

    def probability(self, seq: list) -> np.float:
        """Calculate probability of this path.
        As product of possibilities of each word in seq would be close to 0,
        we use sum of log(possibility(word[i + 1] | word[i])).
        """
        log_prob = 0.
        curr_node = self.root
        for word in seq:
            total = curr_node.count
            if word not in curr_node.child:
                log_prob = 0.
                break
            curr_node = curr_node.child[word]
            log_prob += np.log(curr_node.count / total)
        return log_prob
