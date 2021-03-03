from typing import TypedDict

import cv2
import numpy as np

from services.inference.input_data import InputData
from services.inference.input_data.image import Image
from services.inference.input_data.labels import Labels


class InputDataReader:
    def __init__(self, source):
        self._source = source

    def read(self) -> InputData:
        image = self.read_image()
        labels = self.read_labels()
        return InputData(image=image, labels=labels)

    def read_image(self) -> Image:
        raise NotImplementedError

    def read_labels(self) -> Labels:
        raise NotImplementedError


class RequestLabelData(TypedDict):
    toHairColor: str
    toGender: str
    toYang: bool


class HTTPInputDataReader(InputDataReader):

    def read_image(self) -> Image:
        buffer = self._source.files['image']
        data_as_np_array = np.frombuffer(buffer.read(), np.uint8)
        image_data = cv2.imdecode(data_as_np_array, cv2.IMREAD_COLOR)
        return Image(image_data)

    def read_labels(self) -> Labels:
        request_data = RequestLabelData(self._source.form)
        labels = np.array([0] * 5)
        har_color_index = Labels.hair_colors[request_data['toHairColor']]
        labels[har_color_index] = 1
        labels[3] = Labels.genders[request_data['toGender']]
        labels[4] = int(request_data['toYang'])
        return Labels(labels)


class TGInputDataReader(InputDataReader):
    def __init__(self, source, file_id):
        super().__init__(source)
        self._file_id = file_id

    def read_image(self) -> Image:
        tg_file = self._source.bot.getFile(self._file_id)
        file_as_bytearray = tg_file.download_as_bytearray()
        data_as_np_array = np.frombuffer(file_as_bytearray, np.uint8)
        image_data = cv2.imdecode(data_as_np_array, cv2.IMREAD_COLOR)
        return Image(image_data)

    def read_labels(self) -> Labels:
        labels = np.array([0, 1, 0, 0, 1])
        return Labels(labels)
