from collections import namedtuple
import pandas as pd
from _data_manage import DataManager


class Analyzer(object):
    def __init__(self, manager: DataManager):
        self.__manager = manager
        self.__prob_freq_lower = 4.5  # use k sigma
        self.__prob_ngram_lower = -10.

    def analyze(self, packet: pd.Series or namedtuple) -> dict:
        """Calculate and collect safety factors of packet.
        Entries in factor dict:
        - src / dst: whether the source / destination mac has been seen before
        - connected: whether the two macs have been in a session
        - content_all: whether the content has appeared in global contents
        - content_p2p: whether the content has appeared in current session
        - prob_ngram: probability of content in context using ngram
        - prob_freq: probability of frequency using gaussian estimation

        :param packet: incoming packet
        :return: dict of safety factors
        """
        src, dst, content = packet.src_mac, packet.dst_mac, packet.content
        current_time = packet.time
        manager = self.__manager
        # Factors of safety
        factors = {
            'src': manager.query_mac(src),
            'dst': manager.query_mac(dst),
            'connected': manager.query_connect(src, dst),
            'content_all': manager.query_content(content),
            'content_p2p': manager.query_content(content, (src, dst)),
            'prob_ngram': manager.query_prob_ngram((src, dst), 5),
            'prob_freq': manager.query_prob_freq((src, dst), content,
                                                 current_time)
        }
        return factors

    def judge(self, factors: dict) -> bool:
        """Analyze and evaluate safetiness according to safety factors,
        and decide whether to trust or decline this packet.

        :param factors: evaluated safety factors for current packet
        :return: whether to trust or decline this packet
        """
        if not factors['src'] or not factors['dst']:
            return False
        if not factors['connected'] or not factors['content_p2p']:
            return False
        if not factors['content_all']:
            return False
        if factors['prob_freq'] > self.__prob_freq_lower:
            return False
        if factors['prob_ngram'] < self.__prob_ngram_lower:
            return False
        return True
