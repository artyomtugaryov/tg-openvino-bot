from typing import Dict

import numpy as np

from services.inference.data import Data


class InputImageData(Data):
    @property
    def shape(self) -> np.ndarray:
        return self._data.shape


class InputFlagsData(Data):
    pass


class CompoundInputData(Data):
    def __init__(self, image_data: InputImageData, flags_data: InputFlagsData):
        super(CompoundInputData, self).__init__({
            'image': image_data,
            'flags': flags_data,
        })

    @property
    def shape(self) -> Dict[str, np.ndarray]:
        return {
            'image': self.data['image'].data.shape,
            'flags': self.data['flags'].data.shape
        }

    @property
    def to_infer(self) -> Dict[str, np.ndarray]:
        return {
            'image': self.data['image'].data.data,
            'flags': self.data['flags'].data.data
        }
