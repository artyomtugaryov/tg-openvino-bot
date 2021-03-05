from services.inference.data import InputData


class IReader:
    """
    Interface for classes to read the data from different sources
    """
    def __init__(self, source):
        self._source = source

    def read(self) -> InputData:
        """
        Read data from source and create Data
        :return:
        """
        raise NotImplementedError
