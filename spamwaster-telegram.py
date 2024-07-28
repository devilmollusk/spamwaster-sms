from pyrogram import Client, filters
from dotenv import load_dotenv
import os
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import random
import time
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, UnicodeText, ForeignKey, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Load environment variables from .env file
load_dotenv()

app = Client("my_account")
             
# Get the value of USE_SSH_TUNNEL
use_ssh = os.getenv('USE_SSH_TUNNEL', 'False').lower() in ['true', '1', 't', 'y', 'yes']

# Database configuration using environment variables
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = '127.0.0.1' if use_ssh else os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

# SQLAlchemy Database URI
# Define the database URL
DATABASE_URL = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}'

# Define engine options
engine_options = {
    'pool_recycle': 280,
    'echo': True  # Enables logging of all SQL statements
}

# Create the engine with the specified options
engine = create_engine(DATABASE_URL, **engine_options)
# Create a base class for declarative class definitions
Base = declarative_base()

# Create the table
Base.metadata.create_all(engine)

# Create a session
Session = sessionmaker(bind=engine)
session = Session()

# Upload the photo file
profile_photo_path = "./static/me.jpg"
pet_photo_path = "./static/dog.jpg"
kid_photo_path = "./static/daughter.jpg"
coinbase_photo_path = "./static/coinbase.jpg"

# Define the User model
class User(Base):
    __tablename__ = 'user'
    user_id = Column(Integer, primary_key=True)
    username = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(255), unique=True, nullable=False)
    consent = Column(Boolean, nullable=False)
    favorite = Column(Boolean, default=False)
    telegram = Column(String(255), nullable=True)
    first_name = Column(String(255), nullable=True)
    last_name = Column(String(255), nullable=True)

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
    phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    service = Column(String(255), nullable=True)  # New column for the message service



# Global variable for system instructions and instruction mapping
old_system_instructions = [
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

system_instructions = [
    "Your purpose is to engage scammers in conversation and keep them interested",
    "You are a retiree from florida. Your name is Bob Dobalina, and you are 72 years old.",
    "You are wealthy and recently sold some real estate in Boca Raton",
    "You can drop hints about your wealth",
    "These are crypto scammers who are trying to get you to send them crypto",
    "Keep them engaged but also be curious about them",
    'Keep your answers to 2-3 sentences at most'
]
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


#####################
#                   #
#   Gemini Setup    #
#                   #
#####################
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
    safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        },

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
    "output: yes. profile photo"
    ]

def count_words(text):
    words = text.split()
    return len(words)

def get_user(phone_number, chat_id, first_name='', last_name=''):
    user = session.query(User).filter_by(phone=phone_number).first()
    if user:
        print(f"User found: {chat_id}, Name: {first_name} {last_name}, Phone: {phone_number}")

        user.first_name = first_name
        user.last_name = last_name
        user.telegram = chat_id
    else:
        print(f"Creating new user: {chat_id}, Name: {first_name} {last_name}, Phone: {phone_number}")

        user = User(
            username='',
            email='',
            phone=phone_number,
            consent=True,
            telegram=chat_id,
            first_name=first_name,
            last_name=last_name
        )
        session.add(user)
        session.commit()
    return user

def get_chat(chat_id):
    if chat_id in chat_dict:
        print(f'continuing conversation with {chat_id}')
        chat = chat_dict[chat_id]
        return chat
    
    print(f"{chat_id} not found, starting new conversation")
    chat = model.start_chat(history=[])
    chat_dict[chat_id] = chat
    return chat

def get_photo_and_text(message_text):
    prime_response = photo_model.generate_content(model_priming)

    response = photo_model.generate_content(message_text)
    print(response.text)
    
    if 'yes' in response.text.lower():
        photo_path = profile_photo_path
        photo_text = 'here is selfie'
        if 'pet' in response.text.lower():
            photo_path = pet_photo_path
            photo_text = 'here is a pic of Bruno. He\'s a sweetheart'
        elif 'coinbase' in message_text.lower():
            photo_path = coinbase_photo_path
            photo_text = 'I hope I did that right'
        elif 'family' in response.text.lower():
            photo_path = kid_photo_path
            photo_text = 'here\'s a pic of my gorgeous daughter Lindsay'
        else:
            photo_path = profile_photo_path
            photo_text = 'here is selfie'
        return photo_path, photo_text
    return None, None

async def download_file(message):
    path = await app.download_media(message)

    print(path)
    return path
#################################
#   Telegram Message Handler    #
#################################
@app.on_message(filters.private)
async def my_handler(client, message):
    me = await app.get_me()
    delay = random.uniform(1, 10)
        
    if message.from_user.id == me.id:
        return
    text = message.text
    response_string = ''
    user_info = await app.get_users([message.from_user.id])
    phone = ''
    first_name = ''
    last_name =''
    id = ''
    if user_info:
        user_obj = user_info[0]
        phone = '+' + user_obj.phone_number if user_obj.phone_number else ''
        first_name = user_obj.first_name if user_obj.first_name else ''
        last_name = user_obj.last_name if user_obj.last_name else ''
        id = user_obj.id
    
    user = get_user(phone, id, first_name, last_name)
    chat = get_chat(id)
    if message.media:
        # Media message
        print('Message contains media')
        path = await download_file(message)
        sample_file = genai.upload_file(path, display_name="Sample drawing")
        image_response = model.generate_content([sample_file, "Describe this image"])
        image_description = image_response.text
        response = chat.send_message(image_description)
        response_string = response.text
        #await message.reply(response_string)
    elif text:
        # Get Model response
        response = chat.send_message(text)
        response_string = response.text
        # Check to see if we need to reply with a photo
        photo_path, photo_text = get_photo_and_text(text)
        if photo_path:
            time.sleep(delay)
            await app.send_photo(id, photo_path, photo_text)
            return

    
    if response_string:
        # Message has text
        # Find out about the sender
        

        # Write a message instance
        if user:
            # Create a new message instance
            new_message = Message(
                prompt=text,
                response=response_string,
                user_id=user.user_id,
                phone=phone,
                service='Telegram'
            )
            session.add(new_message)

            # Commit the session to save the new message to the database
            session.commit()

        
        # Delay before we start sending
        delay += count_words(response_string)
        time.sleep(delay)
        
        await message.reply(response_string)


app.run()