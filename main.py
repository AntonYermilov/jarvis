import logging

logging.basicConfig(filename='bot.log', filemode='a', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S')

from telegram.ext import Updater, CommandHandler

from lib.config import BOT_TOKEN, PERSONAL_CHAT_ID, ON_START_STICKER_ID, ON_FAIL_STICKER_ID
from lib.commands import invest_show, invest_suggest, restart_bot
from lib.invest import run_invest_updater


class Bot(object):
    def __init__(self):
        self.updater = Updater(BOT_TOKEN)
        self.updater.dispatcher.add_handler(CommandHandler('invest_show', invest_show))
        self.updater.dispatcher.add_handler(CommandHandler('invest_suggest', invest_suggest))
        self.updater.dispatcher.add_handler(CommandHandler('restart', restart_bot))

    def run(self):
        run_invest_updater()
        self.updater.start_polling()
        self.updater.bot.send_sticker(PERSONAL_CHAT_ID, ON_START_STICKER_ID)
        logging.info('Starting bot')
        try:
            self.updater.idle()
        except:
            self.updater.bot.send_sticker(PERSONAL_CHAT_ID, ON_FAIL_STICKER_ID)


def main():
    bot = Bot()
    bot.run()


if __name__ == '__main__':
    main()
