import io

import cv2
import numpy as np


class Data:
    """
    Abstraction to store data
    """

    def __init__(self, data):
        self.data = data

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


class ImageData(Data):
    @property
    def shape(self) -> np.ndarray:
        return self.data.shape
