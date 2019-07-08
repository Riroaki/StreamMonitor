from collections import namedtuple
import pandas as pd
import numpy as np
from _n_gram import NGram
from _log import logger


class DataManager(object):
    def __init__(self, limit: int = 2000):
        # Params: limit of history entry count
        self.__limit = limit
        self.__count = 0
        # When to update gaussian parameters: half of history is replaced
        self.__update_gauss_limit = self.__limit / 2
        self.__update_count = 0
        # Slots to store extracted information from history
        self.__mac_set = None
        self.__content_encode = None
        self.__content_p2p = None
        self.__session_p2p = None
        self.__ngram_p2p = None
        self.__frequency_p2p = None

    def load(self, data: pd.DataFrame, clear: bool = False):
        # Load history data and extract patterns
        # Clear old history before loading current history
        if clear:
            self.drop()
        # Truncate data if count exceeds limit
        size = len(data)
        if size > self.__limit:
            logger.warning('Load last %d packets to history.' % self.__limit)
        # Extract pattern from data
        self.__mac_set = self.__extract_mac_set(data)
        self.__content_encode = self.__extract_content_encode(data)
        self.__session_p2p = self.__extract_session_p2p(data)
        self.__content_p2p = self.__extract_content_p2p()
        self.__frequency_p2p = self.__extract_frequency_p2p()
        self.__merge_session_p2p()
        self.__ngram_p2p = self.__extract_ngram_p2p()
        logger.debug('Data loaded.')

    @staticmethod
    def __extract_mac_set(data: pd.DataFrame) -> set:
        # Extract macs from data and form a set
        src_macs = set(pd.unique(data.src_mac))
        dst_macs = set(pd.unique(data.dst_mac))
        macs = src_macs.union(dst_macs)
        logger.debug('Extracted mac set.')
        return macs

    @staticmethod
    def __extract_content_encode(data: pd.DataFrame) -> dict:
        # Extract all unique contents and form an encode dict
        content_list = pd.unique(data.content)
        content_dict = {con: i for i, con in enumerate(content_list)}
        logger.debug('Extracted all content encode.')
        return content_dict

    def __extract_session_p2p(self, data: pd.DataFrame) -> dict:
        # Group packages by each (src -> dst) pair
        session_dict = {mac: {} for mac in self.__mac_set}
        session_list = data.groupby(['src_mac', 'dst_mac'])
        for tag, session in session_list:
            src, dst = tag
            session_dict[src][dst] = session
        logger.debug('Extracted sessions for mac pairs.')
        return session_dict

    def __extract_content_p2p(self) -> dict:
        # Extract all unique contents for each (src -> dst) pair
        content_p2p = {mac: {} for mac in self.__mac_set}
        for src, data_flow in self.__session_p2p.items():
            for dst, session in data_flow.items():
                content_p2p[src][dst] = set(pd.unique(session.content))
        logger.debug('Extracted content for mac pairs.')
        return content_p2p

    def __extract_frequency_p2p(self) -> dict:
        # Extarct frequency of each content for each (src -> dst) pair
        frequency_p2p = {mac: {} for mac in self.__mac_set}
        for src, data_flow in self.__session_p2p.items():
            for dst, session in data_flow.items():
                # Extract distribution:
                # mean & variance of frequency for each second
                delta = pd.Timedelta(seconds=1)
                frequency_p2p[src][dst] = {}
                for content in self.__content_p2p[src][dst]:
                    freq_list = []
                    all_time = session.loc[session.content == content].time
                    start_time = session.iloc[0].time
                    last_time = all_time.iloc[-1]
                    last_count = 0
                    while last_time >= start_time:
                        last_time -= delta
                        count = len(all_time.loc[all_time >= last_time])
                        freq_list.append(count - last_count)
                        last_count = count
                    # Get mean and std of frequencies
                    mean = np.mean(freq_list)
                    std = np.std(freq_list)
                    frequency_p2p[src][dst][content] = (mean, std)
        logger.debug('Extracted frequencies for mac pairs.')
        return frequency_p2p

    def __merge_session_p2p(self) -> None:
        # Merge sessions for each (src <-> dst) pair into conversations
        merged_set = set()
        session_dict = self.__session_p2p
        for src, data_flow in session_dict.items():
            for dst, session in data_flow.items():
                # Not merged yet
                if (src, dst) not in merged_set:
                    mutual = session
                    if src in session_dict[dst]:
                        mutual = mutual.append(session_dict[dst][src])
                        # Sort by time and re-index
                        mutual.sort_values('time', ascending=True,
                                           inplace=True)
                        mutual.reset_index(drop=True, inplace=True)
                    # Assign symmetric sides with same reference;
                    # This might be dangerous, as we modify
                    # the dict while iterating on it
                    session_dict[src][dst] = mutual
                    session_dict[dst][src] = mutual
                    # Set dst, src to be merged
                    merged_set.add((dst, src))
        logger.debug('Merged sessions for mac pairs.')

    def __extract_ngram_p2p(self) -> dict:
        # Extract and build n-gram models on each mutual conversation
        ngram_dict = {mac: {} for mac in self.__mac_set}
        for src, data_flow in self.__session_p2p.items():
            for dst, session in data_flow.items():
                if dst not in ngram_dict[src]:
                    ngram = NGram(self.__content_encode, session.content)
                    # Assign symmetric sides with same reference
                    ngram_dict[src][dst] = ngram
                    ngram_dict[dst][src] = ngram
        logger.debug('Extracted bi-gram tables.')
        return ngram_dict

    def update(self, packet: namedtuple) -> None:
        # Update history data by appendinig one entry
        src, dst = packet.src_mac, packet.dst_mac
        content = packet.content
        # Update mac set if necrssary
        if content not in self.__content_encode:
            current_idx = len(self.__content_encode)
            self.__content_encode[content] = current_idx + 1
        # Initialize slots for new mac
        for mac in (src, dst):
            if mac not in self.__mac_set:
                self.__mac_set.add(mac)
                self.__session_p2p[mac] = {}
                self.__content_p2p[mac] = {}
                self.__ngram_p2p[mac] = {}
        # Initialize mutual slots for new connection if both sides are empty
        if dst not in self.__session_p2p[src] \
                and src not in self.__session_p2p[dst]:
            self.__session_p2p[src][dst] = pd.DataFrame()
            self.__content_p2p[src][dst] = set()
            self.__ngram_p2p[src][dst] = NGram(self.__content_encode,
                                               pd.Series([]))
        # Now at least one side is not empty, we assign the symmetric side
        for a, b in [(src, dst), (dst, src)]:
            if a not in self.__session_p2p[b]:
                self.__session_p2p[b][a] = self.__session_p2p[a][b]
                self.__content_p2p[b][a] = self.__content_p2p[a][b]
                self.__ngram_p2p[b][a] = self.__ngram_p2p[a][b]
        # Add current packet information
        row = pd.Series(
            dict(zip(self.__session_p2p[src][dst].columns, list(packet)[1:])))
        ext = self.__session_p2p[a][b].append(row, ignore_index=True)
        self.__session_p2p[a][b] = ext
        self.__content_p2p[a][b].add(packet.content)
        # Update n gram model
        previous = self.__session_p2p[a][b].iloc[-2].content
        self.__ngram_p2p[a][b].update(previous, content)
        # Update gauss model
        self.__update_count += 1
        if self.__update_count >= self.__update_gauss_limit:
            self.__update_count = 0
            self.__frequency_p2p = self.__extract_frequency_p2p()
            logger.debug('Gaussian parameters updated.')
        # Remove oldest entry if current history count exceeds limit
        if self.__count == self.__limit:
            # Reduce oldest connection from n-gram
            old_content_1 = self.__session_p2p[src][dst].iloc[0].content
            old_content_2 = self.__session_p2p[src][dst].iloc[1].content
            self.__ngram_p2p[src][dst].update(old_content_1, old_content_2,
                                              is_remove=True)
            # Remove oldest entry from session
            idx = self.__session_p2p[src][dst].head(1).index
            self.__session_p2p[src][dst].drop(idx, inplace=True)
            extended = self.__session_p2p[src][dst].append(packet)
            self.__session_p2p[src][dst] = extended
        logger.debug('Updated one packet.')

    def drop(self) -> None:
        # Re-initialize information slots
        self.__count = 0
        self.__mac_set = None
        self.__content_encode = None
        self.__content_p2p = None
        self.__session_p2p = None
        self.__ngram_p2p = None
        self.__frequency_p2p = None
        logger.debug('Dropped history data.')

    def query_mac(self, mac: str) -> bool:
        # Check whether mac in current mac set
        exist = mac in self.__mac_set
        return exist

    def query_connect(self, src: str, dst: str) -> bool:
        # Check whether two macs have communicated before
        connected = src in self.__mac_set and dst in self.__session_p2p[src]
        return connected

    def query_content(self, content: str, peers: tuple = None) -> bool:
        # Check whether content exists in previous data:
        # both global session and peer session
        if peers is None:
            exist = content in self.__content_encode
        else:
            src, dst = peers
            try:
                exist = content in self.__content_p2p[src][dst]
            except KeyError:  # src or dst not exist in session
                exist = False
        return exist

    def query_prob_freq(self, peers: tuple, content: str,
                        current_time: pd.Timestamp) -> float:
        # Calculate probability of frequency of current content in session
        src, dst = peers
        try:
            history = self.__session_p2p[src][dst]
            delta = pd.Timedelta(seconds=1)
            previous_time = current_time - delta
            frequency = len(history.loc[(history.time >= previous_time) & (
                    history.content == content)])
            mean, std = self.__frequency_p2p[src][dst][content]
            # Calculate probability in gaussian distribution
            prob = abs(frequency - mean) / std
        except KeyError:
            prob = 0.
        return prob

    def query_prob_ngram(self, peers: tuple, n: int = 5) -> float:
        # Calculate probability of current context using bigram table
        src, dst = peers
        try:
            history = self.__session_p2p[src][dst].tail(n)
            context = history.content.tolist()
            prob = self.__ngram_p2p[src][dst].calc_prob(context)
        except KeyError:
            prob = 0.
        return prob
