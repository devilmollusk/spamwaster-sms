from pyrogram import Client, filters, enums
from pyrogram.types import Message
from pyrogram.errors import RPCError
from dotenv import load_dotenv
import os
import sys
import json
import asyncio
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import random
import time
import pytz
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, Boolean, UnicodeText, ForeignKey, DateTime, or_
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# LLama
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, List
from groq import Groq


# Load environment variables from .env file
load_dotenv()

app = Client("my_account")
my_user = None
starting_directory = os.path.dirname(os.path.abspath(sys.argv[0]))

# Get the value of USE_SSH_TUNNEL
use_ssh = os.getenv('USE_SSH_TUNNEL', 'False').lower() in ['true', '1', 't', 'y', 'yes']

# Database configuration using environment variables
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = '127.0.0.1' if use_ssh else os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

# Use user history when initializing chat
USE_HISTORY =  os.getenv('USE_HISTORY', 'False').lower() in ['true', '1', 't', 'y', 'yes']
USE_DELAY =  os.getenv('USE_DELAY', 'False').lower() in ['true', '1', 't', 'y', 'yes']
AI_MODEL=os.getenv('AI_MODEL')

# Groq AI testing
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY')

LLAMA3_70B_INSTRUCT = "llama3-70b-8192"
LLAMA3_8B_INSTRUCT = "llama3-8b-8192"

DEFAULT_MODEL = LLAMA3_70B_INSTRUCT

llama_client = Groq()
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
photo_dir = "./static/"
profile_photo_path = "./static/me.jpg"
pet_photo_path = "./static/dog.jpg"
kid_photo_path = "./static/daughter.jpg"
coinbase_photo_path = "./static/coinbase.jpg"
bling_photo_path = "./static/crypto_bling.png"
selfie_dict = {
    "selfie_01.jpg": "Here's me enjoying some delicious fruit",
    "selfie_02.jpg": "Here is one of the happiest days of my life: the marriage of my beautiful daughter Nancy. She's a peach!",
    "selfie_03.jpg": "Here's me and my late dog Rufus. I miss him \U0001F622",
    "selfie_04.jpg": "This is me in Puerta Vallarta. Vive Mexico!!!",
    "selfie_05.jpg": "This is me camping in Yosemite",
    "selfie_06.jpg": "Me tending the fire",
    "selfie_07.jpg": "This is me about to hit the trails",
    "selfie_08.jpg": "This is me standing by the ruins of an old bridge",
    "selfie_09.jpg": "This is me enjoying a tasty burger"
}

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
    phone = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    service = Column(String(255), nullable=True)  # New column for the message service
    prompt_media = Column(String(255))
    response_media = Column(String(255))

# Chat session class
class ChatSession:
    def __init__(self):
        self.system_instructions = None
        self.chat_history = []

    def start_chat(self, system_instructions, history=None):
        """Initializes the chat session with system instructions and optional existing history."""
        self.system_instructions = system_instructions
        
        # Prepend system instructions to the history
        self.chat_history = [{"role": "system", "content": system_instructions}]
        
        # Add existing history if provided, excluding any system instructions already there
        if history:
            self.chat_history.extend(history)
            
        # Ensure there's only one system instruction entry
        #self.chat_history = [entry for entry in self.chat_history if entry["role"] != "system"] + [{"role": "system", "content": system_instructions}]
        print(f"Starting LLama chat with: {self.chat_history}")
    def send_message(self, message, role="user"):
        """Sends a message and appends it to the chat history."""
        # Add the user's message to the history
        self.chat_history.append({"role": role, "content": message})

        # Simulate a response from the AI model
        response = self.get_ai_response(self.chat_history)

        # Add the AI's response to the history
        self.chat_history.append({"role": "assistant", "content": response})
        
        return response
    
    def get_ai_response(self, chat_history):
        """Simulates getting a response from an AI model. Replace with actual AI call."""
        # Here, you would implement the logic to communicate with the AI model,
        # passing the chat history and receiving the model's response.
        # For this example, we'll just echo the last user message.
        user_message = chat_history[-1]["content"]
        print(json.dumps(chat_history, indent=4))
        response = chat_completion(chat_history)
        return response
    
# Handle time of day
def get_adjusted_dt(dt):
    est = pytz.timezone('US/Eastern')
    est_time = dt.astimezone(est)
    return est_time

def time_of_day(dt):
    """
    Determine the time of day based on the provided datetime object.
    
    Args:
    dt (datetime): The datetime object to check.
    
    Returns:
    str: One of 'morning', 'noon', 'afternoon', 'evening', or 'late night'.
    """
    hour = dt.hour
    
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 13:
        return 'noon'
    elif 13 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'late night'

def get_last_10_messages(user_id):
    messages = (
        session.query(Message)
        .filter(Message.user_id == user_id)
        .order_by(Message.uuid.asc())
        .all()
    )
    return messages[-10:]
def get_user_history(user):
    messages = get_last_10_messages(user.user_id)
    history=[]
    if 'llama' in AI_MODEL:
        for message in messages:
            user_message = {
                "role": "user",
                "content": message.prompt

            }
            model_message = {
                "role": "assistant",
                "content": message.response
            }
            history.append(user_message)
            history.append(model_message)
    else:
        for message in messages:
            user_message = {
                "role": "user",
                "parts": [
                    message.prompt
                ]
            }
            model_message = {
                "role": "model",
                "parts": [
                    message.response
                ]
            }
            history.append(user_message)
            history.append(model_message)
    return history
    
# Global to store time of day
saved_time_of_day = 'morning'
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

# Define the path to your text file
file_path = 'llama_instructions.txt'

# Read the contents of the file
with open(file_path, 'r') as file:
    llama_instructions = file.read()

print (llama_instructions)
#################
#   LLaMa 3.1   #
#################

# Hugging Face Access Token (replace with your own)
access_token = os.getenv('HUGGING_FACE_ACCESS_TOKEN')

# Helper functions
def assistant(content: str):
    return { "role": "assistant", "content": content }

def llama_user(content: str):
    return { "role": "user", "content": content }

def system(content: str):
    return { "role": "system", "content": content }

# Model ID from Hugging Face Hub
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
def chat_completion(
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
def completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    return chat_completion(
        [llama_user(prompt)],
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

def is_photo(text):
    return chat_completion(
        [system('You are trying to determine if the user input is a request to share a photo over text. Respond with yes or no, along with a determination as to what sort of photo is being requested'),
        llama_user("can you send me a photo of you"),
        assistant("yes. profile photo"),
        llama_user("Can you take a picture of the Coinbase app and send it to me?"),
        assistant("yes. screenshot"),
        llama_user("What sort of dogs do you have?"),
        assistant("no"),
        llama_user("Can you send me a picture of your dogs?"),
        assistant("yes. pet photo"),
        llama_user("Can you show me what you look like?"),
        assistant("yes. profile photo"),
        llama_user("Can you show me what you look like?"),
        assistant("yes. profile photo"),
        llama_user("Where are your photos?"),
        assistant("yes. profile photo"),
        llama_user("Do you have any recent photos of your mom?"),
        assistant("yes. family photo"),
        llama_user("Can you show me what you look like?"),
        assistant("yes. profile photo"),
        llama_user("Show me what you look like"),
        assistant("yes. profile photo"),
        llama_user("What do you look like?"),
        assistant("yes. profile photo"),
        llama_user("What does your mom look like?"),
        assistant("yes. family photo"),
        llama_user("Where are your photos?"),
        assistant("yes. profile photo"),
        llama_user(text),

    ]
    )
'''
# Load tokenizer and model from Hugging Face Hub (requires access token)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=access_token)
model = AutoModelForCausalLM.from_pretrained(model_id, token=access_token)

# Move the model to GPU if available, otherwise CPU
if torch.cuda.is_available():
    model = model.to("cuda")
else:
    model = model.to("cpu")

# Define conversation termination tokens
terminators = [
    tokenizer.eos_token_id,  # End-of-sentence token
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),  # Custom end-of-conversation token
]

# Maximum allowed input token length
MAX_INPUT_TOKEN_LENGTH = 4096

def llama_generate_text(message, history=[], temperature=0.7, max_new_tokens=256, system=""):
    """Generates text based on the given prompt and conversation history.

    Args:
        message: The user's prompt.
        history: A list of tuples containing user and assistant messages.
        temperature: Controls randomness in generation.
        max_new_tokens: Maximum number of tokens to generate.
        system: Optional system prompt.

    Returns:
        The generated text.
    """

    conversation = []
    if system:
        conversation.append({"role": "system", "content": system})

    if history:
        conversation.extend(history)
    conversation.append({"role": "user", "content": message})

    input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt")

    if input_ids.shape[1] > MAX_INPUT_TOKEN_LENGTH:
        input_ids = input_ids[:, -MAX_INPUT_TOKEN_LENGTH:]   


    input_ids = input_ids.to(model.device)   


    generate_kwargs = {
        "input_ids": input_ids,
        "max_length": max_new_tokens + input_ids.shape[1],  # Adjust for total length
        "do_sample": temperature != 0,  # Use sampling for non-zero temperature (randomness)
        "temperature": temperature,
        "eos_token_id": terminators,  # Specify tokens to stop generation
    }

    output = model.generate(**generate_kwargs)[0]
    response = tokenizer.decode(output, skip_special_tokens=True)

    return response
'''
#######################################
def reinitialize_gemini_model(instructions_string):
    utc_time = datetime.now(pytz.utc)
    est_time = get_adjusted_dt(utc_time)
    current_time_of_day = time_of_day(est_time)
    global saved_time_of_day
    new_instructions = instructions_string + [f"Current time of day is {current_time_of_day}"]
    saved_time_of_day = current_time_of_day
    print (f"Time of day is: {saved_time_of_day}")

    model = genai.GenerativeModel(
        model_name=AI_MODEL,
        generation_config=generation_config,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
        },
        system_instruction=new_instructions,
    )
    print(f"Model reinitialized with instructions: {new_instructions}")

    return model


#####################
#                   #
#   Gemini Setup    #
#                   #
#####################

# Cache of chats
chat_dict = {}

# Cache of users
user_dict = {}
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 500,
    "response_mime_type": "text/plain",
}
if 'gemin' in AI_MODEL:
    model = reinitialize_gemini_model(system_instructions)

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
    "input: Where are your photos?",
    "output: yes. profile photo",
    "input: Do you have any recent photos of your mom?",
    "output: yes. family photo",
    "input: can you show me what you look like?",
    "output: yes. profile photo",
    "input: Show me what you look like",
    "output: yes. profile photo",
    "input: What do you look like?",
    "output: yes. profile photo",
    "input: What does your mom look like?",
    "output: yes. family photo",
    "input: Where are your photos?",
    "output: yes. profile photo",
    "input: Show me your wallet",
    "input: yes. screenshot"
    ]

def count_words(text):
    words = text.split()
    return len(words)

def get_user(user_obj):
    phone = '+' + user_obj.phone_number if user_obj.phone_number else None
    first_name = user_obj.first_name if user_obj.first_name else None
    last_name = user_obj.last_name if user_obj.last_name else None
    username = user_obj.username if user_obj.username else None
    id = user_obj.id

    # Try to find the user 
    # Perform the query
    user = session.query(User).filter(
        or_(
            User.phone != '' and User.phone == phone,
            User.username != '' and User.username == username,
            User.telegram != '' and User.telegram == id,
            User.first_name != '' and User.first_name == first_name,
            User.last_name != '' and User.last_name == last_name,
        )
    ).first()

    if user:
        print(f"User found: {id}, Name: {first_name} {last_name}, Phone: {phone}")
        if first_name:
            user.first_name = first_name
        if last_name:
            user.last_name = last_name
        if username:
            user.username = username
        if id:
            user.telegram = id
        if phone:
            user.phone = phone
       
    else:
        print(f"Creating new user: {id}, Name: {first_name} {last_name}, Phone: {phone}")

        user = User(
            username=username,
            email='',
            phone=phone,
            consent=True,
            telegram=id,
            first_name=first_name,
            last_name=last_name
        )
        session.add(user)
    session.commit()
    return user

def get_chat(chat_id):
    # Do we need a new model based on time of day?
    utc_time = datetime.now(pytz.utc)
    est_time = get_adjusted_dt(utc_time)
    current_time_of_day = time_of_day(est_time)
    global model
    global saved_time_of_day
    if chat_id in chat_dict:
        print(f'continuing conversation with {chat_id}')
        chat = chat_dict[chat_id]
    else:
        print(f"{chat_id} not found, starting new conversation")
        history = []
        if USE_HISTORY:
            # Fetch the history for this user
            user = session.query(User).filter(User.telegram==chat_id).first()
            history = get_user_history(user)
            #print(history)
        if 'gemini' in AI_MODEL:    
            chat = model.start_chat(history=history)
        elif 'llama' in AI_MODEL:
            utc_time = datetime.now(pytz.utc)
            est_time = get_adjusted_dt(utc_time)
            current_time_of_day = time_of_day(est_time)
            new_instructions = llama_instructions + f" The current time of day is {current_time_of_day}"
            saved_time_of_day = current_time_of_day
            chat = ChatSession()
            chat.start_chat(new_instructions, history)
        chat_dict[chat_id] = chat

    if current_time_of_day != saved_time_of_day:
        chat = None
        # Need to reinit the model with new time
        if 'gemini' in AI_MODEL:
            model = reinitialize_gemini_model(system_instructions)
            history = chat.history
            chat = model.start_chat(history=history)
        elif 'llama' in AI_MODEL:
            utc_time = datetime.now(pytz.utc)
            est_time = get_adjusted_dt(utc_time)
            current_time_of_day = time_of_day(est_time)
            
            new_instructions = system_instructions + [f"Current time of day is {current_time_of_day}"]
            saved_time_of_day = current_time_of_day
            history = chat.history
            chat = ChatSession()
            chat.start_chat(new_instructions, history)
        chat_dict[chat_id] = chat
    return chat

async def get_user_info(user_id):
    user_info = None
    if user_id in user_dict:
        user_info = user_dict[user_id]
    else:
        user_info = await app.get_users([user_id])

    return user_info

def get_photo_and_text(message_text):
    response_text = ''
    if 'gemini' in AI_MODEL:
        prime_response = photo_model.generate_content(model_priming)

        response = photo_model.generate_content(f"is this asking for a photo, and if so what sort? {message_text}")
        response_text = response.text
        
    elif 'llama' in AI_MODEL:
        response_text = is_photo(message_text)
    print(f"asking for a photo? {response_text}")
    
    if 'yes' in response_text.lower():
        photo_path = profile_photo_path
        photo_text = 'here is selfie'
        if 'pet' in response_text.lower():
            photo_path = pet_photo_path
            photo_text = 'here is a pic of Bruno. He\'s a sweetheart'
        elif 'coinbase' in message_text.lower():
            photo_path = coinbase_photo_path
            photo_text = 'I hope I did that right'
        elif 'screenshot' in response_text.lower():
            photo_path = bling_photo_path
            photo_text = 'Here\'s what I have at the moment in crypto. I\'ve got about a million more in other investments'
        elif 'family' in response_text.lower():
            photo_path = kid_photo_path
            photo_text = 'here\'s a pic of my gorgeous daughter Lindsay'
        else:
            # Select a random key-value pair
            random_key_value_pair = random.choice(list(selfie_dict.items()))

            # Unpacking the key-value pair
            photo, photo_text = random_key_value_pair
            photo_path = photo_dir + photo
        return photo_path, photo_text
    return None, None

async def download_file(message):
    path = await app.download_media(message)

    print(path)
    return path
#################################
#   Telegram Message Handler    #
#################################
@app.on_message()
async def my_handler(client, message):
    if message.outgoing:
        return
    print(f"OnMessage handler: {client} \n{message}")
    global my_user
    if my_user == None:
        my_user = await app.get_me()
    id = message.from_user.id
    delay = random.uniform(1, 10)
    relative_path = ''
    
    if message.from_user.id == my_user.id:
        return
    text = message.text
    response_string = ''
    user_info = await get_user_info(id)
    if user_info:
        user_obj = user_info[0]
        id = user_obj.id
        user = get_user(user_obj)
    chat = get_chat(id)
    if message.media and message.media == enums.MessageMediaType.PHOTO:
        # Media message
        print(f'Message contains media: ')
        path = await download_file(message)
        relative_path = os.path.relpath(path, starting_directory)
        sample_file = genai.upload_file(path, display_name="Sample drawing")
        image_response = model.generate_content([sample_file, "Describe this image"])
        image_description = image_response.text
        response = chat.send_message(image_description)
        if 'llama' in AI_MODEL:
            response_string = response
        else:
            response_string = response.text
        #await message.reply(response_string)
    elif text:
        # Get current time for EST
        utc_time = datetime.now(pytz.utc)
        est = pytz.timezone('US/Eastern')
        est_time = utc_time.astimezone(est)

        # Get Model response
        response = chat.send_message(text)
        if 'llama' in AI_MODEL:
            response_string = response
        else:
            response_string = response.text

        # Check to see if we need to reply with a photo
        photo_path, photo_text = get_photo_and_text(text)
        if photo_path:
            if user:
                # Create a new message instance
                new_message = Message(
                    prompt=text,
                    response=f"You sent a photo and this text: {photo_text}",
                    user_id=user.user_id,
                    phone=user.phone,
                    service='Telegram',
                    response_media=photo_path,
                    prompt_media=relative_path
                )
                session.add(new_message)

                # Commit the session to save the new message to the database
                session.commit()
            if USE_DELAY:
                time.sleep(delay)
            try:
                await app.send_photo(id, photo_path, photo_text)
                chat.chat_history.append(llama_user(text))
                chat.chat_history.append(system(f"You sent a photo and this text: {photo_text}"))
            except RPCError as e:
                print(f"error sending photo {e}")
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
                phone=user.phone,
                service='Telegram',
                prompt_media=relative_path
            )
            session.add(new_message)

            # Commit the session to save the new message to the database
            session.commit()

        
        # Delay before we start sending
        delay += count_words(response_string)
        if USE_DELAY:
            time.sleep(delay)
        
        
        await message.reply(response_string)

app.run()