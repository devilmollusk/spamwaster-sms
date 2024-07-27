from pyrogram import Client
import os
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_API_ID=os.getenv('TELEGRAPH_APP_ID')
TELEGRAM_API_HASH=os.getenv('TELEGRAPH_API_HASH')
TELEGRAM_PHONE=os.getenv('TELEGRAPH_PHONE')

app = Client("my_account", api_id=TELEGRAM_API_ID, api_hash=TELEGRAM_API_HASH)

app.run()