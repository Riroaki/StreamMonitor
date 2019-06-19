from collections import namedtuple
import pandas as pd
import logging
from trie import Trie


class Model(object):
    def __init__(self):
        """Init of model, defining entry slots."""
        self.__mac_all = None
        self.__content_all = None
        self.__session_p2p = None
        self.__content_p2p = None
        self.__mode_p2p = None
        # Constants in relation to context, safety, and ...
        self.__history_limit = 1000
        self.__history_count = 0
        self.__context_size = 100
        self.__handlers = {True: self.__handle_safe,
                           False: self.__handle_unsafe}
        self.__prob_bound = -1.0
        self.__freq_bound = 0.3
        self.__tolerance = 0.1
        self.__unsafe_macs = set()

    def load_history(self, data: pd.DataFrame,
                     clear: bool = True,
                     limit: int = 1000) -> None:
        """Load history data into model for predictions."""
        if clear:
            self.clear_history()
        assert limit > 0
        count = len(data)
        self.__history_limit = limit
        self.__history_count = min(count, limit)
        # Truncate data if its size exceeds limit
        if count > limit:
            start_idx = data.head(1).index[0] + count - limit
            data = data.loc[start_idx:]
            logging.warning('Truncated last %d history packets.' % len(data))
        # Extract information from data
        self.__mac_all = self.__extract_mac(data)
        self.__content_all = self.__extract_content_all(data)
        self.__session_p2p = self.__extract_session_p2p(data)
        self.__content_p2p = self.__extract_content_p2p()
        self.__mode_p2p = self.__extract_mode_p2p()

    def clear_history(self) -> None:
        """Clear data load from history."""
        self.__mac_all = None
        self.__content_all = None
        self.__session_p2p = None
        self.__content_p2p = None
        self.__mode_p2p = None
        self.__unsafe_macs = set()
        self.__history_count = 0

    def process(self, packet: pd.Series or namedtuple) -> None:
        """Evaluate the safetiness of an incoming packet and
        perform corresponding operations after decided whether to trust
        or decline this packet.

        :param packet: incoming packet
        """
        safety_factors = self.__calc_factors(packet)
        is_safe = self.__analyze_safety(safety_factors)
        self.__handlers[is_safe](packet, safety_factors)

    @staticmethod
    def __extract_mac(data: pd.DataFrame) -> set:
        """Extract mac address set from history data."""
        src_macs = set(pd.unique(data.src_mac))
        dst_macs = set(pd.unique(data.dst_mac))
        all_macs = src_macs.union(dst_macs)
        return all_macs

    @staticmethod
    def __extract_content_all(data: pd.DataFrame) -> dict:
        """Extract all content and form encode dict (content to index)
        from history data.
        """
        content_list = pd.unique(data.content)
        content_dict = {con: i for i, con in enumerate(content_list)}
        return content_dict

    def __extract_session_p2p(self, data: pd.DataFrame) -> dict:
        """Extract all sessions between two mac addresses from history data.
        - session_dict[src][dst] tracks packets from src mac to dst mac;
        - as a mutual relation, session_dict[src][dst] share content with
            session_dict[dst][src].
        """
        mac_set = self.__mac_all
        session_dict = {mac: {} for mac in mac_set}
        # Group sessions between macs
        data_flow = data.groupby(['src_mac', 'dst_mac'])
        for flow in data_flow:
            src, dst = flow[0]
            session_dict[src][dst] = flow[1]
        # Merge sessions between 2 macs into a complete conversation
        merged = set()
        for src, sessions in session_dict.items():
            for dst in sessions.keys():
                # Already merged
                if (src, dst) in merged:
                    continue
                # Record mac pair that has been merged
                merged.add((dst, src))
                # If other side doesn't exist
                if src not in session_dict[dst]:
                    session_dict[dst][src] = session_dict[src][dst]
                    continue
                # Merge two sessions into one
                mutual = session_dict[src][dst].append(session_dict[dst][src])
                # Sort by time and re-index
                mutual.sort_values('time', ascending=True, inplace=True)
                mutual.reset_index(drop=True, inplace=True)
                # Assign symmetric sides with same reference
                session_dict[src][dst] = mutual
                session_dict[dst][src] = mutual
        return session_dict

    def __extract_content_p2p(self) -> dict:
        """Extract content codes from session data of each machine
            between two mac addresses.
        - content_dict[src][dst] tracks packets from src mac to dst mac;
        - content_dict[src][dst] does NOT share content with
            content_dict[dst][src].
        """
        mac_set = self.__mac_all
        session_dict = self.__session_p2p
        content_dict = {mac: {} for mac in mac_set}
        # Same shape of session_dict
        for src, sessions in session_dict.items():
            for dst, flow in sessions.items():
                # Get contents appeared in conversation flow
                content_dict[src][dst] = set(
                    flow.loc[flow.src_mac == src].content.unique())
        return content_dict

    def __extract_mode_p2p(self) -> dict:
        """Extract mode from sessions between two mac addresses.
        A mode is a repeated sequence of contents, stored in a Trie tree.
        A mode always starts with a request content packet.
        - mode_dict[src][dst] tracks mode between src mac and dst mode;
        - as a mutual relation, mode_dict[src][dst] shares content with
            mode_dict[dst][src].
        """
        session_dict = self.__session_p2p
        mode_dict = {mac: {} for mac in self.__mac_all}
        # Same shape of session dict and content dict
        for src, sessions in session_dict.items():
            for dst, flow in sessions.items():
                # Skip if already extracted
                if dst in mode_dict[src]:
                    continue
                # Get request check list
                is_request = flow.content.apply(self.__is_request).tolist()
                # Add a dummy tail
                is_request.append(True)
                mode_count, curr_seq, idx = {}, [], 0
                for row in flow.itertuples():
                    content = row.content
                    curr_seq.append(self.__content_all[content])
                    # End of sequence: request content
                    if not is_request[idx] and is_request[idx + 1]:
                        seq = tuple(curr_seq)
                        mode_count[seq] = mode_count.get(seq, 0) + 1
                        curr_seq = []
                    idx += 1
                # Build Trie tree with mode dict
                trie = Trie()
                for mode, count in mode_count.items():
                    trie.insert(mode, count)
                # Assign symmetric sides with same reference
                mode_dict[src][dst] = trie
                mode_dict[dst][src] = trie
        return mode_dict

    def __calc_factors(self, packet: pd.Series or namedtuple) -> dict:
        """Calculate and collect safety factors of packet.
        Entries in factor dict:
        - src / dst: whether the source / destination mac has been seen before
        - friend: whether the two macs have been in a session
        - content_all: whether the content has appeared in global contents
        - content_p2p: whether the content has appeared in current session
        - prob: probability of content in context and current session's mode

        :param packet: incoming packet
        :return: dict of safety factors
        """

        def __get_context() -> list:
            """Get encoded context content list of current packet.
            Context is defined as the last several packets of current session.
            """
            _df = self.__session_p2p[src][dst].tail(self.__context_size)
            _df.append(packet)
            _con_series = _df.content
            _context = _con_series.tolist()
            return _context

        def __freq_request() -> float:
            """Assess probability of request packet in current context.
            Use frequency of current request to represent probability.
            """
            _freq = context.count(content) / len(context)
            return _freq

        def __prob_response() -> tuple:
            """Assess probability of response packet in current context.
            Use posterior probability of current sequence
            to represent probability.
            """
            _seq = [content]
            for _con in context[::-1]:
                if self.__is_request(_con):
                    _seq.append(_con)
            _seq.reverse()
            # Calculate probability of current sequency using mode trie.
            _trie = self.__mode_p2p[src][dst]
            _prob = _trie.probability(_seq)
            return _seq, _prob

        src, dst, content = packet.src_mac, packet.dst_mac, packet.content
        # Factors of safety
        factors = {
            'src': src in self.__mac_all,
            'dst': dst in self.__mac_all,
            'content_all': content in self.__content_all,
            'is_request': self.__is_request(content),
            'friend': src in self.__mac_all and dst in self.__session_p2p[src],
            'content_p2p': False,
            'freq': 0.,
            'prob': 0.,
            'seq': []
        }
        # Whether the content of this packet has appeared in
        # the conversation between src and dst machines before
        if factors['friend']:
            factors['content_p2p'] = content in self.__content_p2p[src][dst]
        # Calculate probability for this packet, according to history patterns
        if factors['content_p2p']:
            # Calculate probability for request / response packet
            context = __get_context()
            if factors['is_request']:
                factors['freq'] = __freq_request()
            else:
                factors['seq'], factors['prob'] = __prob_response()
        return factors

    def __analyze_safety(self, factors: dict) -> bool:
        """Analyze and evaluate safetiness according to safety factors,
        and decide whether to trust or decline this packet.

        :param factors: evaluated safety factors for current packet
        :return: whether to trust or decline this packet
        """
        if not factors['src'] or not factors['dst']:
            return False
        if not factors['content_all']:
            return False
        if not factors['friend'] or not factors['content_p2p']:
            return False
        # TODO: deal with frequency and probability
        if factors['is_request']:
            if abs(factors['freq'] - self.__freq_bound) > self.__tolerance:
                return False
            if factors['prob'] < self.__prob_bound:
                return False
        return True

    @staticmethod
    def __is_request(content: str) -> bool:
        """Check the type of content: request / response."""
        res = content.find('request') >= 0
        return res

    def __handle_safe(self, packet: pd.Series or namedtuple,
                      factors: dict) -> None:
        """Handle safe packets: update history with this packet."""
        logging.info('Trust packet:', packet)
        logging.info('Factors:', factors)
        self.__update_history(packet)
        self.__update_mode(packet, factors)

    def __handle_unsafe(self, packet: pd.Series or namedtuple,
                        factors: dict) -> None:
        """Handle unsafe packets: report this packet and its seriousness."""
        logging.warning('Untrust packet:', packet)
        logging.warning('Factors:', factors)
        self.__unsafe_macs.add(packet.src_mac)

    def __update_history(self, packet: pd.Series or namedtuple) -> None:
        """Update one row into history."""
        src, dst = packet.src_mac, packet.dst_mac
        # Drop oldest packet and append new one
        session = self.__session_p2p[src][dst]
        if self.__history_count >= self.__history_limit:
            session.drop(session.head(1).index, inplace=True)
        else:
            self.__history_count += 1
        session.append(packet)

    def __update_mode(self, packet: pd.Series or namedtuple,
                      factors: dict) -> None:
        """Update mode trie."""
        src, dst = packet.src_mac, packet.dst_mac
        # Update mode trie
        trie = self.__mode_p2p[src][dst]
        seq = factors['seq']
        trie.insert(seq)
