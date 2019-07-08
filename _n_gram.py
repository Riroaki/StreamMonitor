import pandas as pd
import numpy as np


class NGram(object):
    def __init__(self, encode_dict: dict, series: pd.Series):
        # Initialize bigram table and encode dict slots
        self.__encode = encode_dict
        # Build bigram table for series of words
        words = series.unique()
        bigram = {encode_dict[word]: {} for word in words}
        encoded = series.apply(lambda x: encode_dict[x]).tolist()
        for i in range(1, len(encoded)):
            first, second = encoded[i - 1], encoded[i]
            bigram[first][second] = bigram[first].get(second, 0) + 1
        self.__bigram = bigram

    def calc_prob(self, seq: list) -> np.float:
        # Calculate log posterior probability for a sequence
        prob = 0.
        encoded = [self.__encode[word] for word in seq]
        # Calculate sum of log posterior probability
        for i in range(1, len(seq)):
            first, second = encoded[i - 1], encoded[i]
            if first not in self.__bigram:
                # TODO
                prob -= 1
            elif second not in self.__bigram[first]:
                prob -= 1
            else:
                total = sum(self.__bigram[first].values())
                posterior = self.__bigram[first][second] / total
                prob += np.log(posterior)
        return prob

    def update(self, previous: object, current: object,
               is_remove: bool = False) -> None:
        previous, current = self.__encode[previous], self.__encode[current]
        if not is_remove:
            # Register encoded words if they were not in the bigram table
            if previous not in self.__bigram:
                self.__bigram[previous] = {}
            # Update by adding 1 count
            if current not in self.__bigram[previous]:
                self.__bigram[previous][current] = 0
            self.__bigram[previous][current] += 1
        else:
            # Reduce oldest connection
            self.__bigram[previous][current] -= 1
