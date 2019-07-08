from collections import namedtuple
import pandas as pd
from _log import logger
from _analyze import Analyzer
from _data_manage import DataManager


class Monitor(object):
    def __init__(self, manager: DataManager):
        """Init of model, defining entry slots."""
        self.__manager = manager
        self.__analyzer = Analyzer(manager)
        # Constants in relation to context, safety, and ...
        self.__handlers = {True: self.__handle_safe,
                           False: self.__handle_unsafe}
        self.__count = {'safe': 0, 'unsafe': 0}
        # self.__unsafe_macs = set()

    def process(self, packet: pd.Series or namedtuple) -> bool:
        """Evaluate the safetiness of an incoming packet and
        perform corresponding operations after decided whether to trust
        or decline this packet.
        TODO: add multi-thread support if necessary

        :param packet: incoming packet
        :return: whether this packet is safe
        """
        safety_factors = self.__analyzer.analyze(packet)
        is_safe = self.__analyzer.judge(safety_factors)
        self.__handlers[is_safe](packet, safety_factors)
        return is_safe

    def __handle_safe(self, packet: pd.Series or namedtuple,
                      factors: dict) -> None:
        """Handle safe packets: update history with this packet."""
        logger.info('Trust packet.')
        logger.info('Factors: %s' % str(factors))
        self.__manager.update(packet)
        self.__count['safe'] += 1

    def __handle_unsafe(self, packet: pd.Series or namedtuple,
                        factors: dict) -> None:
        """Handle unsafe packets: report this packet and its seriousness."""
        logger.warning('Untrust packet.')
        logger.warning('Factors: %s' % str(factors))
        self.__count['unsafe'] += 1
