from typing import List, Type

from services.inference.data import Data


class IDataProcessor:
    """
    Interface for any (pre and post) data processing
    """

    def __init__(self, data: Data):
        self._data = data

    def process(self) -> Data:
        raise NotImplementedError


class DataProcessPipeline:
    def __init__(self, data_processors: List[Type[IDataProcessor]]):
        self._data_processors = data_processors

    def run(self, data: Data) -> Data:
        processed_data = data
        for data_processor_class in self._data_processors:
            data_processor = data_processor_class(processed_data)
            processed_data = data_processor.process()
        return processed_data
