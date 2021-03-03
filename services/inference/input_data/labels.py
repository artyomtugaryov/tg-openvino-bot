import numpy as np


class Labels:
    hair_colors = {
        'black': 0,
        'blond': 1,
        'brow': 2
    }
    genders = {
        'female': 0,
        'male': 1,
    }

    def __init__(self, data: np.ndarray):
        self._data = data

    def prepare(self) -> np.ndarray:
        labels_data = np.ndarray(shape=(1, *self._data.shape))
        labels_data[0] = self._data
        return labels_data
