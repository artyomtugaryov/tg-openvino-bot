import io
from enum import Enum
from typing import Any, Dict

import cv2
import numpy as np

from services.inference.input_utils import InputType


class Data:
    """
    Abstraction to store N tensors data
    """

    def __init__(self, data):
        self.data = data

    @property
    def shape(self) -> Any:
        return self.data.shape

    @property
    def _as_uint8_array(self) -> np.ndarray:
        return self.data.astype(np.uint8)

    def to_image(self, image_path: str):
        """
        Save data as image
        :param image_path: path to result image
        """
        cv2.imwrite(image_path, self.data)

    def to_file_object(self) -> io.BytesIO:
        """
        Create file like object from data
        :return: file-like object contains data
        """
        is_success, buffer = cv2.imencode('.bmp', self.data)

        # create file-object in memory
        buffer_storage = io.BytesIO(buffer)

        cv2.imdecode(np.frombuffer(buffer_storage.getbuffer(), np.uint8), -1)
        return buffer_storage


class InputData:
    """
    Class to manipulate of input data fro Engine
    """

    def __init__(self):
        self._data: Dict[InputType, Data] = {}

    def add_input(self, input_type: InputType, data: Data):
        self._data[input_type] = data

    @property
    def shape(self) -> Dict[InputType, np.ndarray]:
        return {
            input_type: data.shape
            for input_type, data in self._data.items()
        }

    @property
    def raw_data(self) -> Dict[InputType, np.ndarray]:
        return {
            input_type: data.data
            for input_type, data in self._data.items()
        }


class InferenceResultData:
    """
    Class to manipulate of inference result data from Engine
    """

    def __init__(self, inference_results: Dict[str, np.ndarray]):
        self._data: Dict[Enum, Data] = self._parse_output_data(inference_results)

    def _parse_output_data(self, something) -> Dict[str, np.ndarray]:
        raise NotImplementedError

    @property
    def shape(self) -> Dict[Enum, np.ndarray]:
        return {
            input_type: data.shape
            for input_type, data in self._data.items()
        }

    @property
    def raw_data(self) -> Dict[InputType, np.ndarray]:
        return {
            input_type: data.data
            for input_type, data in self._data.items()
        }
