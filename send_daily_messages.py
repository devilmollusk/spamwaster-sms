from pyrogram import Client, filters, enums
from pyrogram.types import Message
from pyrogram.errors import RPCError
from dotenv import load_dotenv
import os
import sys
import json
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import random
import time
from datetime import datetime, timedelta

import pytz
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, UnicodeText, ForeignKey, DateTime, or_, func
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
# LLama
from typing import Dict, List
from groq import Groq

# Load environment variables from .env file
load_dotenv()

app = Client("my_account")

# Database configuration using environment variables
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

# Use user history when initializing chat
USE_HISTORY =  os.getenv('USE_HISTORY', 'False').lower() in ['true', '1', 't', 'y', 'yes']
USE_DELAY =  os.getenv('USE_DELAY', 'False').lower() in ['true', '1', 't', 'y', 'yes']
USE_HOSTED_LLAMA = os.getenv('USE_HOSTED_LLAMA', 'False').lower() in ['true', '1', 't', 'y', 'yes']
AI_MODEL=os.getenv('AI_MODEL')

# Groq AI testing
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

LLAMA3_70B_INSTRUCT = os.getenv('LLAMA3_70B_INSTRUCT')
LLAMA3_8B_INSTRUCT = os.getenv('LLAMA3_8B_INSTRUCT')

DEFAULT_MODEL = LLAMA3_70B_INSTRUCT
LLAMA_URL = os.getenv('LLAMA_URL')
LLAMA_MODEL = os.getenv('LLAMA_MODEL')

llama_client = Groq()
# Define the path to your text file
file_path = 'llama_instructions.txt'
# Read the contents of the file
with open(file_path, 'r') as file:
    llama_instructions = file.read()

# SQLAlchemy Database URI
# Define the database URL
DATABASE_URL = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}'

# Define engine options
engine_options = {
    'pool_recycle': 280,
    'echo': False  # Enables logging of all SQL statements
}

# Create the engine with the specified options
engine = create_engine(DATABASE_URL, **engine_options)
# Create a base class for declarative class definitions
Base = declarative_base()

# Create a session
Session = sessionmaker(bind=engine)
session = Session()
session.rollback()

# Define the User model
class User(Base):
    __tablename__ = 'user'
    user_id = Column(Integer, primary_key=True)
    username = Column(String(255), nullable=False)
    email = Column(String(255), nullable=True)
    phone = Column(String(255), nullable=True)
    consent = Column(Boolean, nullable=False)
    favorite = Column(Boolean, default=False)
    telegram = Column(String(255), nullable=True)
    first_name = Column(UnicodeText, nullable=True)
    last_name = Column(UnicodeText, nullable=True)

    def __init__(self, username, email, phone, consent, telegram, first_name, last_name):
        self.username = username
        self.email = email
        self.phone = phone
        self.consent = consent
        self.telegram = telegram
        self.first_name = first_name
        self.last_name = last_name

class Message(Base):
    __tablename__ = 'messages'
    uuid = Column(Integer, primary_key=True, autoincrement=True)
    prompt = Column(UnicodeText, nullable=False)
    response = Column(UnicodeText, nullable=False)
    user_id = Column(Integer, ForeignKey('user.user_id'), nullable=False)
    phone = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    service = Column(String(255), nullable=True)  # New column for the message service
    prompt_media = Column(String(255))
    response_media = Column(String(255))

class History(Base):
    __tablename__='history'
    uuid = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey('user.user_id'), nullable=False)
    role = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    text = Column(UnicodeText, nullable=False)
    media = Column(String(255), nullable=True)

# Create the table 
Base.metadata.create_all(engine)

# Handle time of day
def get_adjusted_dt(dt):
    est = pytz.timezone('US/Eastern')
    est_time = dt.astimezone(est)
    return est_time

def get_last_10_messages(user_id):
    messages = (
        session.query(History)
        .filter(History.user_id == user_id)
        .order_by(History.uuid.asc())
        .all()
    )
    return messages[-10:]
def get_user_history(user_id):
    messages = get_last_10_messages(user_id)
    history=[]
    ai_role = "assistant" if "llama" in AI_MODEL else "model"
    
    for message in messages:
        text = message.text
        role = "user" if message.role == "user" else ai_role
        new_message = {
            "role": role,
            "content": text
        }
        
        history.append(new_message)
    
    return history

# Define the path to your text file
file_path = 'llama_instructions.txt'

# Read the contents of the file
with open(file_path, 'r') as file:
    llama_instructions = file.read()

# Helper functions
def assistant(content: str):
    return { "role": "assistant", "content": content }

def assistant_json(content: json):
    return { "role": "assisntant", "content": content}

def llama_user(content: str):
    return { "role": "user", "content": content }

def system(content: str):
    return { "role": "system", "content": content }

async def chat_completion(
        messages: List[Dict],
        model = DEFAULT_MODEL,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        
        response = llama_client.chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content

        
def get_users_with_last_message(session):
    # This query fetches each user with the timestamp of their most recent message
    subquery = session.query(
        History.user_id,
        func.max(History.created_at).label('last_message_time')
    ).group_by(History.user_id).subquery()

    users_with_last_message = session.query(
        User.user_id,
        User.telegram,
        subquery.c.last_message_time
    ).join(subquery, User.user_id == subquery.c.user_id).all()

    return users_with_last_message
def add_history(role: str, text:str, user_id:str, media:str = ''):
    new_history = History(
        role=role,
        text=text,
        user_id=user_id,
        media=media
    )
    session.add(new_history)
    session.commit()

async def send_message_to_user(telegram_id, user_id, message):
    print(f"sending message to {telegram_id}:\n{message}")
    try:
        
        await app.send_message(telegram_id, message)
        
        add_history('ai', message, user_id)
    except RPCError as e:
        print(e)

async def check_and_send_messages():
    now = datetime.utcnow()
    currrent_time = get_adjusted_dt(now)
    system_instructions = llama_instructions + f"\nThe current time in Florida is {currrent_time}"

    await app.start()
    users = get_users_with_last_message(session)

    for user_id, telegram, last_message_time in users:
        time_since_last_message = now - last_message_time
        minutes_since_last_message = time_since_last_message.total_seconds() / 60
        hours_since_last_message = minutes_since_last_message / 60
        days_since_last_message = hours_since_last_message / 24
        if days_since_last_message > 1:

            messages = get_user_history(user_id)

            messages.append(system(system_instructions))
            messages.append(llama_user(f"\nIt\'s been {days_since_last_message} days since you messaged with this user. Please take into accoutn the last message sent by the user, and please craft a message to restart the conversation"))

            response = await chat_completion(messages)
            await send_message_to_user(telegram, user_id, response)
    await app.stop()
async def main():
    await check_and_send_messages()

app.run(main())

    