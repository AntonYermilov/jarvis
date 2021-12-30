import os
import sys
import logging

import telegram
from telegram import Update
from telegram.ext import CallbackContext

from .invest import investments


def invest_show(update: Update, context: CallbackContext):
    user = update.message.from_user
    user_name, user_id = str(user['username']).lower(), str(user['id'])
    logging.info(f'/invest_show from {user_name}')

    if user_name != 'antonyermilov':
        update.message.reply_text('Only @antonyermilov is allowed to use this command!')
        return

    try:
        logging.info('Updating spreadsheet')
        spreadsheet_url = investments.update_spreadsheet()

        logging.info('Visualizing history')
        img = investments.visualize_history()

        update.message.reply_photo(photo=img, caption=f'<b>Spreadsheet URL</b>: <a href="{spreadsheet_url}">link</a>',
                                   parse_mode=telegram.ParseMode.HTML)
    except Exception as e:
        update.message.reply_text(f'<b>Error</b>: {e}', parse_mode=telegram.ParseMode.HTML)


def invest_suggest(update: Update, context: CallbackContext):
    user = update.message.from_user
    user_name, user_id = str(user['username']).lower(), str(user['id'])
    logging.info(f'/invest_suggest from {user_name}')

    if user_name != 'antonyermilov':
        update.message.reply_text('Only @antonyermilov is allowed to use this command!')
        return

    if len(context.args) != 1 or not context.args[0].isdigit():
        update.message.reply_text('Expected exactly one argument: available budget in dollars')
        return

    budget = int(context.args[0])

    try:
        media = []
        for i in range(10):
            suggestions = investments.suggest_shares(budget)
            suggestions.sort()
            ema_visualization = investments.visualize_ema(suggestions)
            suggestions_text = ', '.join(suggestions)
            media.append(telegram.InputMediaPhoto(media=ema_visualization, caption=f'{i}. {suggestions_text}'))
        update.message.reply_media_group(media=media)
    except Exception as e:
        update.message.reply_text(f'<b>Error</b>: {e}', parse_mode=telegram.ParseMode.HTML)


def restart_bot(update: Update, context: CallbackContext) -> None:
    user = update.message.from_user
    user_name, user_id = str(user['username']).lower(), str(user['id'])
    logging.info(f'/restart from {user_name}')

    if user_name != 'antonyermilov':
        update.message.reply_text('Only @antonyermilov is allowed to use this command!')
        return

    update.message.reply_text('Restarting bot')
    python = sys.executable
    os.execl(python, python, *sys.argv)
