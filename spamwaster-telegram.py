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
from typing import Dict, List
from groq import Groq
from huggingface_hub import login


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
USE_HOSTED_LLAMA = os.getenv('USE_HOSTED_LLAMA', 'False').lower() in ['true', '1', 't', 'y', 'yes']
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

# Create a session
Session = sessionmaker(bind=engine)
session = Session()
session.rollback()

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
        self.chat_history = [entry for entry in self.chat_history if entry["role"] != "system"] + [{"role": "system", "content": system_instructions}]
        #print(f"Starting LLama chat with: {self.chat_history}")
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
        #print (user_message)
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
        session.query(History)
        .filter(History.user_id == user_id)
        .order_by(History.uuid.asc())
        .all()
    )
    return messages[-10:]
def get_user_history(user):
    messages = get_last_10_messages(user.user_id)
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

def assistant_json(content: json):
    return { "role": "assisntant", "content": content}

def llama_user(content: str):
    return { "role": "user", "content": content }

def system(content: str):
    return { "role": "system", "content": content }

def construct_prompt(examples, new_input=None):
    """
    Constructs a prompt for an LLM with given examples and a new input.

    :param examples: List of tuples, each containing (input_text, output_dict).
    :param new_input: Optional new input string for which to generate a response.
    :return: Constructed prompt string.
    """
    prompt = "The following are examples of inputs and their corresponding JSON outputs:\n\n"
    for input_text, output_dict in examples:
        prompt += f"Input: {input_text}\n"
        prompt += f"Output: {json.dumps(output_dict)}\n\n"

    if new_input:
        prompt += f"Input: {new_input}\n"
        prompt += "Output: "

    return prompt

# Model ID from Hugging Face Hub
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
def chat_completion(
        messages: List[Dict],
        model = DEFAULT_MODEL,
        temperature: float = 0.6,
        top_p: float = 0.9,
    ) -> str:
        if USE_HOSTED_LLAMA:
            response = llama_client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                top_p=top_p,
            )
        else:
            llama_generate_text(messages)
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
    system_prompt = """
        You are trying to determine if the user input is a request to share a photo over text. 
        Respond with yes or no, along with a determination as to what sort of photo is being requested. 
        Return in JSON format like:
         {
            "is_photo": bool,
            "photo_type": str
         }
    """
    return chat_completion(
        [system(system_prompt),
        llama_user("can you send me a photo of you"),
        assistant(" { \"is_photo\": \"yes\", \"profile_photo\": \"profile photo\" }"),
    
        llama_user(text),

    ]
    )

if not USE_HOSTED_LLAMA:
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

def llama_generate_text(messages, temperature=0.7, max_new_tokens=256, system=""):
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

    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

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

processed_messages = []
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 500,
    "response_mime_type": "text/plain",
}
photo_config = {
    "temperature": 0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 500,
    "response_mime_type": "text/plain",
}

# TODO: need to use LLama for photo eval
photo_eval_model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=photo_config,
        safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
            },

        system_instruction="your role is to evaluate an image and describe what the image depicts in 2 sentences",
    )
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

        system_instruction="""
            Your job it to determine whether the prompt is a request to share a photo
        """
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
    phone = '+' + user_obj.phone_number if user_obj.phone_number else ''
    first_name = user_obj.first_name if user_obj.first_name else ''
    last_name = user_obj.last_name if user_obj.last_name else ''
    username = user_obj.username if user_obj.username else ''
    id = user_obj.id

    # Try to find the user 
    # Perform the query
    user = session.query(User).filter(
        or_(
            User.username != '' and User.username == username,
            User.telegram != '' and User.telegram == id,
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
    chat = None
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
            print (new_instructions)
            saved_time_of_day = current_time_of_day
            chat = ChatSession()
            chat.start_chat(new_instructions, history)
        chat_dict[chat_id] = chat

    if current_time_of_day != saved_time_of_day:
        
        # Need to reinit the model with new time
        if 'gemini' in AI_MODEL:
            model = reinitialize_gemini_model(system_instructions)
            history = chat.history
            chat = model.start_chat(history=history)
        elif 'llama' in AI_MODEL:
            utc_time = datetime.now(pytz.utc)
            est_time = get_adjusted_dt(utc_time)
            current_time_of_day = time_of_day(est_time)
            
            new_instructions = system_instructions + f"Current time of day is {current_time_of_day}"
            saved_time_of_day = current_time_of_day
            history = chat.chat_history
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
        print (response_text)
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

def add_history(role: str, text:str, user:User, media:str = ''):
    new_history = History(
        role=role,
        text=text,
        user_id=user.user_id,
        media=media
    )
    session.add(new_history)
    session.commit()

async def download_file(message):
    path = await app.download_media(message)

    print(path)
    return path
#################################
#   Telegram Message Handler    #
#################################
@app.on_message(filters.text | filters.photo)
async def my_handler(client, message):
    if message.outgoing or message.id in processed_messages:
        return
    print(f"OnMessage handler: {client} \n{message}")
    processed_messages.append(message.id)
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
        image_response = photo_eval_model.generate_content([sample_file, "Describe this image"])
        image_description = image_response.text
        model_input = f"An image was sent with this description: {image_description}"
        response = chat.send_message(model_input)
        if 'llama' in AI_MODEL:
            response_string = response
        else:
            response_string = response.text
        text = model_input
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
                response_text = f"You sent a photo and this text: {photo_text}"
                add_history('user', text, user, relative_path)
                
            if USE_DELAY:
                time.sleep(delay)
            try:
                await app.send_photo(id, photo_path, photo_text)
                add_history('ai', response_text, user, photo_path)

                chat.chat_history.append(llama_user(text))
                chat.chat_history.append(assistant(f"You sent a photo and this text: {photo_text}"))
            except RPCError as e:
                print(f"error sending photo {e}")
            return

    add_history('user', text, user, relative_path)
    if response_string:
        
        # Delay before we start sending
        delay += count_words(response_string)
        if USE_DELAY:
            time.sleep(delay)
        
        if user:
            add_history('ai', response_string, user)
        
        await message.reply(response_string)

app.run()