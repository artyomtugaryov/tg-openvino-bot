from enum import Enum

from services.inference import InputInfo


class GenderInputLayout(Enum):
    nchw = 'NCHW'
    nc = 'NC'


class GenderInputType(Enum):
    image = 'image'
    flags = 'flags'
    undefined = 'undefined'


class GenderInputInfo(InputInfo):
    @staticmethod
    def define_input_type(input_info_data) -> GenderInputType:
        if input_info_data.layout == GenderInputLayout.nc.value:
            return GenderInputType.flags
        if input_info_data.layout == GenderInputLayout.nchw.value:
            return GenderInputType.image
        return GenderInputType.undefined
