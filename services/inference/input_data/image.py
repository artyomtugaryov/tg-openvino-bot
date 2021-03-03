import cv2
import numpy as np


class Image:
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    def __init__(self, data: np.ndarray):
        self._data = data

    def prepare(self, ):
        resized_image = self._resize(data=self._data)
        rgb_image = resized_image[:, :, ::-1]  # BGR to RGB
        chw_rgb_image = rgb_image.transpose((2, 0, 1))  # HWC to CHW
        normalized_image = chw_rgb_image / chw_rgb_image.max()
        for channel in range(normalized_image.shape[0]):
            normalized_image[channel] = (normalized_image[channel] - self.mean[channel]) / self.std[channel]

        input_data = np.ndarray(shape=(1, *normalized_image.shape))
        input_data[0] = normalized_image
        return input_data

    @staticmethod
    def _resize(data: np.ndarray) -> np.ndarray:
        height = 300
        # initialize the dimensions of the image to be resized and
        # grab the image size
        h, w = data.shape[:2]

        # check to see if the width is None

        r = height / float(h)
        dim = (int(w * r), height)

        # resize the image
        resized = cv2.resize(data, dim, interpolation=cv2.INTER_AREA)

        # return the resized image
        return resized
