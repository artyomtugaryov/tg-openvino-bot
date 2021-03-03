import logging
from enum import Enum

from telegram.ext import Updater, CommandHandler, MessageHandler, Filters, ConversationHandler

from services.inference.input_data.reader import TGInputDataReader
from services.inference.service import GenderInferenceService

token = '1641535882:AAFlqkjNQ74mpEPPSCpCdUV0U8UHUV5_Jv0'

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
    print(1)
    if not context.user_data.get(States.start_over.value):
        update.message.reply_text(
            'Hi, I\'m OpenVINO Bot and I can magik. Just send me photo!')

    context.user_data[States.start_over.value] = False
    return States.sending_photo.value


def stop(update, context):
    """End Conversation by command."""
    update.message.reply_text('Okay, bye.')

    return States.end


def get_image(update, context):
    infer_result = TGInputDataReader(context, update.message.photo[-1].file_id).read()\
        .using(GenderInferenceService).infer().prepare_to_send()

    context.bot.send_photo(chat_id=update.effective_chat.id, photo=infer_result)


def main():
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging.INFO)

    updater = Updater(token=token, use_context=True)
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler('start', start)],

        states={
            States.sending_photo.value: [MessageHandler(Filters.photo, get_image)],
        },

        fallbacks=[CommandHandler('stop', stop)],
    )
    updater.dispatcher.add_handler(conv_handler)
    updater.start_polling()


if __name__ == '__main__':
    main()
