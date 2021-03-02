import cv2
import numpy as np


class InputData:
    def __init__(self, request):
        self.request_data = request
        data_from_request = self.request_data.files['image'].read()
        data_as_np_array = np.frombuffer(data_from_request, np.uint8)
        self.image = cv2.imdecode(data_as_np_array, cv2.IMREAD_COLOR)

    def prepare(self, shape) -> np.ndarray:
        image = cv2.resize(self.image, shape[-2:])
        image = image.transpose((2, 0, 1))

        input_data = np.ndarray(shape=shape)
        input_data[0] = image

        return input_data
