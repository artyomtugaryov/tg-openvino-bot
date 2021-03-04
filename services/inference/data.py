import io

from PIL import Image
import numpy as np
import cv2

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
        img = Image.fromarray(self._as_uint8_array)

        # create file-object in memory
        file_object = io.BytesIO()

        # write PNG in file-object
        img.save(file_object, 'PNG')

        # move to beginning of file so `send_file()` it will read from start
        file_object.seek(0)

        return file_object
