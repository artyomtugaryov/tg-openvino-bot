import logging
import os
from enum import Enum

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler

from bot.data_reader import TelegramImageReader
from services.inference import DataProcessPipeline
from services.inference.face_detection import (FaceDetectionInputData, ImageResizePreProcessor,
                                               ImageBGRToRGBPreProcessor,
                                               FaceDetectionEngine, ImageHWCToCHWPreProcessor,
                                               ExpandDimImagePreProcessor)
from services.inference.face_detection.data_processor import FaceDetectionPostProcessor, DrawFaceBoxes
from services.inference.input_utils import InputType

token = os.environ.get('TELEGRAM_TOKEN')

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)


class States(Enum):
    sending_photo = 'sending_photo'
    start_over = 'start_over'
    end = 'end'


def start(update, context):
    if not context.user_data.get(States.start_over.value):
        update.message.reply_text('Hi, I am OpenVINO Bot and I can do magic. Just send me a photo!')

    context.user_data[States.start_over.value] = False
    return States.sending_photo.value


def stop(update, context):
    """End Conversation by command."""
    update.message.reply_text('Okay, bye.')

    return States.end.value


engine = FaceDetectionEngine()


def process_image(update, context):
    file_id = update.message.photo[-1].file_id
    image_data = TelegramImageReader(source=context, file_id=file_id).read()

    processed_image_data = DataProcessPipeline(
        [
            ImageResizePreProcessor(),
            ImageBGRToRGBPreProcessor(),
            ImageHWCToCHWPreProcessor(),
            ExpandDimImagePreProcessor(),
        ]
    ).run(image_data)

    full_data = FaceDetectionInputData(image_data=processed_image_data)

    inference_result = engine.infer(full_data)
    input_shape = engine.input_shape()[InputType.image]
    processed_results = DataProcessPipeline(
        [
            FaceDetectionPostProcessor(image_data.shape, input_shape),
            DrawFaceBoxes(image_data),
        ]
    ).run(inference_result)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=processed_results.to_file_object())


def main():
    updater = Updater(token=token, use_context=True)
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler('start', start)
        ],

        states={
            States.sending_photo.value: [MessageHandler(Filters.photo, process_image)],
        },

        fallbacks=[
            CommandHandler('stop', stop)
        ],
    )
    updater.dispatcher.add_handler(conv_handler)
    updater.start_polling()


if __name__ == '__main__':
    main()
