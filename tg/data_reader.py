import numpy as np
import cv2

from services.inference.gender.data import InputImageData, InputFlagsData
from services.inference.reader import IReader


class TelegramImageReader(IReader):
    def __init__(self, source, file_id):
        super().__init__(source)
        self._file_id = file_id

    def read(self) -> InputImageData:
        tg_file = self._source.bot.getFile(self._file_id)
        file_as_bytearray = tg_file.download_as_bytearray()
        data_as_np_array = np.frombuffer(file_as_bytearray, np.uint8)
        image_data = cv2.imdecode(data_as_np_array, cv2.IMREAD_COLOR)
        return InputImageData(image_data)


class TelegramFlagsReader(IReader):
    def __init__(self, source, file_id):
        super().__init__(source)
        self._file_id = file_id

    def read(self) -> InputFlagsData:
        return InputFlagsData(np.array([0, 1, 0, 0, 1]))
