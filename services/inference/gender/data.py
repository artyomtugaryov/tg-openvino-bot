from services.inference import Data
from services.inference.data import InputData
from services.inference.input_utils import InputType


class GenderInputData(InputData):
    def __init__(self,
                 image_data: Data,
                 flags_data: Data):
        super().__init__()
        self.add_input(input_type=InputType.image, data=image_data)
        self.add_input(input_type=InputType.tensor, data=flags_data)
