from PIL import Image
import io


class InferenceResult:
    def __init__(self, data):
        self.data = data

    def prepare_to_send(self) -> io.BytesIO:
        img = Image.fromarray(self.data.astype('uint8'))

        # create file-object in memory
        file_object = io.BytesIO()

        # write PNG in file-object
        img.save(file_object, 'PNG')

        # move to beginning of file so `send_file()` it will read from start
        file_object.seek(0)

        return file_object
