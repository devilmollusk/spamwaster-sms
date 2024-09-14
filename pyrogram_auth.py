from pyrogram import Client
import os
from dotenv import load_dotenv
import argparse

load_dotenv()

TELEGRAM_API_ID=os.getenv('TELEGRAPH_APP_ID')
TELEGRAM_API_HASH=os.getenv('TELEGRAPH_API_HASH')
TELEGRAM_PHONE=os.getenv('TELEGRAPH_PHONE')

parser = argparse.ArgumentParser(description="add session name arg")
parser.add_argument('--session', type=str, required=True, help="The name of the session")
args = parser.parse_args()

app = Client(args.session, api_id=TELEGRAM_API_ID, api_hash=TELEGRAM_API_HASH)

app.run()