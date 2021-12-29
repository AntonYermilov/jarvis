import json


with open('secrets.json', 'r') as fp:
    _secrets = json.load(fp)


TINKOFF_API_TOKEN = _secrets.get('tinkoff_api')
GOOGLE_SA_CONFIG = _secrets.get('google_sa_config')
BOT_TOKEN = _secrets.get('telegram_bot')

PERSONAL_CHAT_ID = 47917044
ON_START_STICKER_ID = 'CAACAgQAAxkBAAECBClgS4S6SqnwD5eGDI5JTI7ouqRmSwACSAADL9_4CZeyLLwowLY4HgQ'
ON_FAIL_STICKER_ID = 'CAACAgQAAxkBAAECXmZgt6yg-TsZJ13SL9MgK5O-4AxzkwACawADL9_4CTzfu7L-2cnNHwQ'
