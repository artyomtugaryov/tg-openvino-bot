from services.inference.data import Data


class IReader:
    """
    Interface for classes to read the data from different sources
    """
    def __init__(self, source):
        self._source = source

    def read(self) -> Data:
        """
        Read data from source and create Data
        :return:
        """
        raise NotImplementedError
