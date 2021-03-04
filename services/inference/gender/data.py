from typing import Dict

import numpy as np

from services.inference import Data, ImageData, InputSchema
from services.inference.gender.input_schema import GenderInputType


class InputFlagsData(Data):
    @property
    def shape(self) -> np.ndarray:
        return self.data.shape


class GenderInputData(Data):
    def __init__(self,
                 image_data: ImageData,
                 flags_data: InputFlagsData,
                 input_schema: InputSchema):
        super(GenderInputData, self).__init__({
            input_schema.layer_name_by_type(GenderInputType.image): image_data,
            input_schema.layer_name_by_type(GenderInputType.flags): flags_data,
        })
        self._input_schema = input_schema

    @property
    def shape(self) -> Dict[str, np.ndarray]:
        return {
            input_name: data.shape
            for input_name, data in self.data.items()
        }

    @property
    def to_infer(self) -> Dict[str, np.ndarray]:
        return {
            input_name: data.data
            for input_name, data in self.data.items()
        }
