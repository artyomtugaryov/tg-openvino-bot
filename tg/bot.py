import logging
import os
from enum import Enum

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler

from services.inference import DataProcessPipeline
from services.inference.gender import (GenderInputData, ImageResizePreProcessor, ImageBGRToRGBPreProcessor,
                                       ImageHWCToCHWPreProcessor, ExpandShapePreProcessor, GenderPostProcessor,
                                       ImageNormalizePreProcessor, GenderEngine)
from tg.data_reader import TelegramImageReader, TelegramFlagsReader

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
        update.message.reply_text(
            'Hi, I am OpenVINO Bot and I can do magic. Just send me a photo!')

    context.user_data[States.start_over.value] = False
    return States.sending_photo.value


def stop(update, context):
    """End Conversation by command."""
    update.message.reply_text('Okay, bye.')

    return States.end


def process_image(update, context):
    file_id = update.message.photo[-1].file_id
    image_data = TelegramImageReader(source=context, file_id=file_id).read()
    flags_data = TelegramFlagsReader(source=context, file_id=file_id).read()

    processed_image_data = DataProcessPipeline([ImageResizePreProcessor(),
                                                ImageBGRToRGBPreProcessor(),
                                                ImageHWCToCHWPreProcessor(),
                                                ImageNormalizePreProcessor(),
                                                ExpandShapePreProcessor()
                                                ]).run(image_data)
    processed_flags_data = DataProcessPipeline([ExpandShapePreProcessor()]).run(flags_data)

    full_data = GenderInputData(image_data=processed_image_data,
                                flags_data=processed_flags_data)

    inference_result = GenderEngine().infer(full_data)

    processed_results = DataProcessPipeline([GenderPostProcessor(),
                                             ImageResizePreProcessor(512),
                                             ImageBGRToRGBPreProcessor()
                                             ]).run(inference_result)
    context.bot.send_photo(chat_id=update.effective_chat.id, photo=processed_results.to_file_object())


def main():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    updater = Updater(token=token, use_context=True)
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            States.sending_photo.value: [MessageHandler(Filters.photo, process_image)],
        },

        fallbacks=[CommandHandler('stop', stop)],
    )
    updater.dispatcher.add_handler(conv_handler)
    updater.start_polling()


if __name__ == '__main__':
    main()
