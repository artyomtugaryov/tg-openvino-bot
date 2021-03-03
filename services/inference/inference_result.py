from PIL import Image
import io
import numpy as np


class InferenceResult:
    def __init__(self, data: np.ndarray):
        self._data = data

    def post_processed(self) -> np.ndarray:
        result = ((np.moveaxis(self._data[0], [0], [2]) + 1) / 2)[:, ::-1, ::-1]
        result *= 255
        return result

    def prepare_to_send(self) -> io.BytesIO:
        data = self.post_processed()
        data = data[:, :, ::-1].copy()
        img = Image.fromarray(data.astype('uint8'))

        # create file-object in memory
        file_object = io.BytesIO()

        # write PNG in file-object
        img.save(file_object, 'PNG')

        # move to beginning of file so `send_file()` it will read from start
        file_object.seek(0)

        return file_object
