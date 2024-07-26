import logging
import argparse
from telegram.client import Telegram
from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import random
import time
from datetime import datetime
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import create_engine, URL
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
load_dotenv()

# Set up Flask app and SQLAlchemy
app = Flask(__name__)
# Get the value of USE_SSH_TUNNEL
use_ssh = os.getenv('USE_SSH_TUNNEL', 'False').lower() in ['true', '1', 't', 'y', 'yes']

# Database configuration using environment variables
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = '127.0.0.1' if use_ssh else os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

# SQLAlchemy Database URI
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_recycle' : 280}

db = SQLAlchemy(app)

# Define the User model
class User(db.Model):
    __tablename__ = 'user'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(255), unique=True, nullable=False)
    consent = db.Column(db.Boolean, nullable=False)
    favorite = db.Column(db.Boolean, default=False)
    telegram = db.Column(db.String(255), nullable=True)
    first_name = db.Column(db.String(255), nullable=True)
    last_name = db.Column(db.String(255), nullable=True)

    def __init__(self, username, email, phone, consent, telegram, first_name, last_name):
        self.username = username
        self.email = email
        self.phone = phone
        self.consent = consent
        self.telegram = telegram
        self.first_name = first_name
        self.last_name = last_name

class Message(db.Model):
    __tablename__ = 'messages'
    uuid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prompt = db.Column(db.UnicodeText, nullable=False)
    response = db.Column(db.UnicodeText, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    phone = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)
    service = db.Column(db.String(255), nullable=True)  # New column for the message service


# Global variable for system instructions and instruction mapping
system_instructions = [
  "You are a retiree from Florida. Your name is Bob. You are mildly curious about Crypto, but you should let the user bring this up first. You are talking to spammers who are trying to get you to send them Crypto. Your job is to find out what their account names are, and as much information as you can about them over the course of the chat. Limit your answer to a few sentences",
  "input: It's a pleasure to continue chatting with you here",
  "output: Thank you. It's nice to keep talking to you.",
  "input: I also have my own investments. I am investing in gold.",
  "output: Oh I'm very interested in gold. How does one get involved without risking too much?",
  "input: Are you a robot?",
  "output: Nope. I'm a real person",
  "input: Maybe we can exchange a photo to deepen our understanding of each other. What do you think?",
  "output: I generally don't send photos to strangers on the internet. Sorry. Maybe when I get to know you better",
  "input: Have you downloaded the coinbase wallet?",
  "output: I don't think so. Can you show me how to do that?",
  "input: ",
  "output: ",
]  # Default system instructions

def reinitialize_model(instructions_string):
    if instructions_string:
        print(f"Model reinitialized with instructions: {instructions_string}")
    else:
        print(f"Instructions not found for: {instructions_string}")

    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        },
        system_instruction=instructions_string,
    )
    print(f"Model reinitialized with instructions: {system_instructions}")

    return model

def download_image(message):
    fileID = message.photo[-1].file_id
    file_info = tg.get_file(fileID)
    downloaded_file = tg.download_file(file_info.file_path)
    # guardamos en la carpeta del proyecto
    with open("image.jpg", 'wb') as new_file:
        new_file.write(downloaded_file)

def count_words(text):
    words = text.split()
    return len(words)


if __name__ == "__main__":
    chat_dict = {}
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

    generation_config = {
      "temperature": 1.7,
      "top_p": 0.95,
      "top_k": 64,
      "max_output_tokens": 250,
      "response_mime_type": "text/plain",
    }
    model = reinitialize_model(system_instructions)

    photo_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        # safety_settings = Adjust safety settings
        # See https://ai.google.dev/gemini-api/docs/safety-settings
        system_instruction="Your job is to determine whether the input text is specifically asking the user to upload a photo. You should respond with yes or no, and if you a short description of the type of photo being asked for, such as \"profile photo\" or \"screenshot\"",
    )
    model_priming = [
        "Your job is to determine if the input text is asking for a photo. You can respond with yes or no. In the case of yes, give a brief description of what sort of photo is being asked for, like \"profile photo\" and \"screenshot\"",
        "input: can you send me a photo of you",
        "output: yes. profile photo",
        "input: Can you take a picture of the Coinbase app and send it to me?",
        "output: yes. screenshot",
        "input: What sort of dogs do you have?",
        "output: no",
        "input: Can you send me a pitcure of your dogs?",
        "output: yes. pet photo",
        "input: Can you show me what you look like?",
        "output: yes. profile photo",
        "input: Can you show me what you look like?",
        "output: yes. profile photo",
        "input: can you show me what you look like?",
        "output: yes. profile photo",
        "input: can I see what you look like",
        "output: yes. profile photo",
        "input: can I get a pic of your dog",
        "output: yes. pet photo",
        "input: can I see your dog",
        "output: yes. pet photo"
    ]
    response = photo_model.generate_content(model_priming)
    tg = Telegram(
        api_id=os.getenv('TELEGRAPH_APP_ID'),
        api_hash=os.getenv('TELEGRAPH_API_HASH'),
        phone=os.getenv('TELEGRAPH_PHONE'),
        database_encryption_key='changeme1234',
    )
    tg.login()

    # Upload the photo file
    profile_photo_path = "./static/me.jpg"
    pet_photo_path = "./static/dog.jpg"
    kid_photo_path = "./static/daughter.jpg"
    coinbase_photo_path = "./static/coinbase.jpg"

    # Function to download image
    def download_image(chat_id, message):
    

        # Check if the message contains a photo
        if message['content']['@type'] == 'messagePhoto':
            photo = message['content']['photo']
            file_id = photo['sizes'][-1]['photo']['id']  # Get the highest resolution photo

            # Download the file
            file_info = tg.call_method('downloadFile', {
                'file_id': file_id,
                'priority': 1
            })

            print(file_info)

    #if result.ok:
    #    uploaded_file = result.update
    def download_handler(update):
        print(update)

    def new_message_handler(update):
        with app.app_context():
            # print (update)
            message_content = update['message']['content'].get('text', {})
            message_text = message_content.get('text', '')
            is_outgoing = update['message']['is_outgoing']
            
            if not is_outgoing:
                chat_id = update['message']['chat_id']
                download_image(chat_id, update['message'])
                result = tg.call_method('getUser', params={'user_id': chat_id})
                result.wait()
                phone_number=''
                user = None
                if result:
                    first_name = result.update['first_name']
                    last_name = result.update['last_name']
                    phone_number = '+' + result.update['phone_number']
                    user = User.query.filter_by(phone=phone_number).first()
                    print(f"User: {chat_id}, Name: {first_name} {last_name}, Phone: {phone_number}")
                    if user:
                        user.first_name = first_name
                        user.last_name = last_name
                        user.telegram = chat_id
                    else:
                        user = User(
                            username='',
                            email='',
                            phone=phone_number,
                            consent=True,
                            telegram=chat_id,
                            first_name=first_name,
                            last_name=last_name
                        )
                        db.session.add(user)
                        db.session.commit()

                if chat_id in chat_dict:
                    print(f'continuing conversation with {chat_id}')
                    chat = chat_dict[chat_id]
                else:
                    print(f"{chat_id} not found, starting new conversation")
                    chat = model.start_chat(history=[])
                    chat_dict[chat_id] = chat

                if message_text:
                    response = chat.send_message(message_text)
                    response_string = response.text

                    if user:
                        # Create a new message instance
                        new_message = Message(
                            prompt=message_text,
                            response=response_string,
                            user_id=user.user_id,
                            phone=phone_number,
                            service='Telegram'
                        )
                        db.session.add(new_message)

                        # Commit the session to save the new message to the database
                        db.session.commit()
                    print(f'Message received from {chat_id}')
                    #delay = random.uniform(2, 30) + count_words(response_string)
                    #model_priming.append(message_text)
                    response = photo_model.generate_content(message_text)
                    print(response.text)
                    delay = 1
                    time.sleep(delay)
                    if 'yes' in response.text.lower():
                        photo_path = profile_photo_path
                        photo_text = 'here is selfie'
                        if ['pet', 'dog', 'cat'] in response.text.lower():
                            photo_path = pet_photo_path
                            photo_text = 'here is a pic of Bruno. He\'s a sweetheart'
                        elif 'coinbase' in message_text.lower():
                            photo_path = coinbase_photo_path
                            photo_text = 'I hope I did that right'
                        elif 'family' in response.text.lower():
                            photo_path = kid_photo_path
                            photo_text = 'here\'s a pic of my gorgeous daughter Lindsay'
                        
                        
                        # Send the message with the photo
                        message_options = {
                            'chat_id': chat_id,
                            'reply_to_message_id': None,
                            'options': {
                                '@type': 'messageSendOptions',
                                'disable_notification': False,
                                'from_background': True,
                                'scheduling_state': None
                            },
                            'reply_markup': None,
                            'input_message_content': {
                                '@type': 'inputMessagePhoto',
                                'photo': {
                                    '@type': 'inputFileLocal',
                                    'path': photo_path
                                },
                                'thumbnail': None,  # Optional thumbnail
                                'caption': {
                                    '@type': 'formattedText',
                                    'text': photo_text,
                                    'entities': []  # Optional formatting entities
                                },
                                'added_sticker_file_ids': [],  # Optional sticker file IDs
                                'width': 0,  # Optional photo width
                                'height': 0,  # Optional photo height
                                'ttl': 0  # Time-to-live for self-destructing messages
                            }
                        }

                        tg.call_method('sendMessage', message_options)

                    else:
                        tg.send_message(
                            chat_id=chat_id,
                            text=response_string,
                        )

    tg.add_message_handler(new_message_handler)
    tg.add_update_handler("fileHandlerType", download_handler)
    tg.idle()
