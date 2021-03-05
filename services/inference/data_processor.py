from typing import List

from services.inference.data import Data


class IDataProcessor:
    """
    Interface for any (pre and post) data processing
    """

    def process(self, data: Data) -> Data:
        raise NotImplementedError


class DataProcessPipeline:
    def __init__(self, data_processors: List[IDataProcessor]):
        self._data_processors = data_processors

    def run(self, data: Data) -> Data:
        processed_data = data
        for data_processor in self._data_processors:
            processed_data = data_processor.process(processed_data)
        return processed_data
