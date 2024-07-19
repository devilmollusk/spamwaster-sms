from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
#from flask_socketio import SocketIO, emit
import requests
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
from dotenv import load_dotenv
import vonage
from pprint import pprint
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
from twilio.http.http_client import TwilioHttpClient
import logging
import mysql.connector
from flask_sqlalchemy import SQLAlchemy
import re
from datetime import datetime
import time
import random

#import sshtunnel

#sshtunnel.SSH_TIMEOUT = 15.0
#sshtunnel.TUNNEL_TIMEOUT = 15.0

# Load environment variables from .env file
load_dotenv()
# Get the value of USE_SSH_TUNNEL
use_ssh = os.getenv('USE_SSH_TUNNEL', 'False').lower() in ['true', '1', 't', 'y', 'yes']

app = Flask(__name__)
#socketio = SocketIO(app)

logging.basicConfig(level=logging.INFO)

app.chat_dict = {}
app.mode_dict = {}

#proxy_client = TwilioHttpClient(proxy={'http': os.environ['http_proxy'], 'https': os.environ['https_proxy']})

VONAGE_API_KEY = os.getenv("VONAGE_API_KEY")
VONAGE_API_SECRET = os.getenv("VONAGE_API_SECRET")
current_dir = os.path.dirname(os.path.realpath(__file__))
keys_dir = os.path.join(current_dir, 'keys')
VONAGE_APPLICATION_PRIVATE_KEY_PATH = os.path.join(keys_dir, 'private.key')
VONAGE_APPLICATION_ID = '63614fd4-31ea-4085-aeb4-9d5719ae1029'

TWILIO_ACCOUNT_SID = os.environ["TWILIO_ACCOUNT_SID"]
TWILIO_AUTH_TOKEN = os.environ["TWILIO_AUTH_TOKEN"]
POSTACK_API_KEY = os.environ['POSTACK_API_KEY']

REQUIRE_CONSENT = os.environ["REQUIRE_CONSENT"]

# Database configuration using environment variables
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')
db_host = '127.0.0.1' if use_ssh else os.getenv('DB_HOST')
db_name = os.getenv('DB_NAME')

# SQLAlchemy Database URI
app.config['SECRET_KEY'] = 'ZL1ZPkd3GKvJ8YShocvUgVSkzLQcaCo7PQRa/C/AAdU='
app.config['SQLALCHEMY_DATABASE_URI'] = f'mysql+mysqlconnector://{db_user}:{db_password}@{db_host}/{db_name}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Global variable for system instructions and instruction mapping
system_instructions = 'Your purpose is to waste the time of SMS scammers. Act interested and engage in conversation. Pretend to be a real person. You are communicating over text'  # Default system instructions
instruction_mapping = {
    'Shy': 'You are a shy teenager communicating over text.',
    'Confident': 'You are a confident speaker with assertive communication.',
    'Grouchy': 'You are feeling grouchy today and your responses may be short.',
    'Distracted': 'You are distracted and your responses may not be fully focused.'
}

# Database configuration
db_config = {
    'user': 'devilmollusk',
    'password':'test',
    'host': 'devilmollusk.mysql.pythonanywhere-services.com',
    'database': 'devilmollusk$starter-db'
}


API_KEY = 'sk-proj-crMCYjp7L3abdVGmPPIzT3BlbkFJSIJUXb83rIH8jqPGbAWu'
API_URL = 'https://api.openai.com/v1/chat/completions'

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {API_KEY}'
}

history = []
# Initialize the database
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_recycle' : 280}
db = SQLAlchemy(app)

# Define the User and Message models
class User(db.Model):
    __tablename__ = 'user'
    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(255), nullable=False)
    phone = db.Column(db.String(255), unique=True, nullable=False)
    consent = db.Column(db.Boolean, nullable=False)

    def __init__(self, username, email, phone, consent):
        self.username = username
        self.email = email
        self.phone = phone
        self.consent = consent
        
class Message(db.Model):
    __tablename__ = 'messages'
    uuid = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prompt = db.Column(db.String, nullable=False)
    response = db.Column(db.String, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.user_id'), nullable=False)
    phone = db.Column(db.String, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow, nullable=False)

# Function to strip non-numeric characters from phone number
def clean_phone_number(phone_number):
    return re.sub(r'\D', '', phone_number)

# Create the database and the User table
with app.app_context():
    db.create_all()

# Function to get database connection
def get_db_connection():

    print('attempting to connect to database')
    connection = mysql.connector.connect(**db_config)
    return connection

# Function to reinitialize Gemini AI model with new instructions
def reinitialize_model(instructions_string):
    
    if instructions_string:
        print(f"Model reinitialized with instructions: {instructions_string}")
        # Pass instructions_string to your model initialization or processing logic
        # Example:
        # model.initialize(instructions_string)
    else:
        print(f"Instructions not found for: {instructions_string}")
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        #safety_settings = BLOCK_NONE,
        safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE
        },
        system_instruction=instructions_string,
    )
    print(f"Model reinitialized with instructions: {system_instructions}")

    return model
    
# Function to run once before handling the first request
with app.app_context():
    # Add your initialization code here
    print("Flask server is initializing...")

    # Initialize GenAI configuration with API key from environment variable
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


    # Create the modelt
    #TODO: need to update to support multiple users

    # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
    generation_config = {
      "temperature": 1.7,
      "top_p": 0.95,
      "top_k": 64,
      "max_output_tokens": 150,
      "response_mime_type": "text/plain",
    }
    app.client = vonage.Client(
        application_id=VONAGE_APPLICATION_ID,
        private_key=VONAGE_APPLICATION_PRIVATE_KEY_PATH,
        key=VONAGE_API_KEY,
        secret=VONAGE_API_SECRET
    )
    app.model = reinitialize_model(system_instructions)
    app.chat = app.model.start_chat(history=history)

# Function to handle consent
def handle_consent(phone_number, consent=True):

    user = User.query.filter_by(phone=phone_number).first()
    if user:
        user.consent = consent
        db.session.commit()
    else:
        # Add a new user
        new_user = User(
            username='',
            email='',
            phone=phone_number,
            consent=consent
        )
        db.session.add(new_user)
        db.session.commit()
        user = new_user
    
    return user

def has_consent(phone_number):
    user = User.query.filter_by(phone=phone_number).first()
    if user:
        return user.consent
    else:
        return False

def postack_send_message(msg, to_phone, from_phone):
    url = "https://api.postack.dev/v1/messages/sms"

    payload = {
        "from": from_phone,
        "to": to_phone,
        "text": msg
    }
    headers = {
        "Authorization": f"Bearer {POSTACK_API_KEY}",
        "Content-Type": "application/json"
    }
    print(headers)
    response = requests.request("POST", url, json=payload, headers=headers)

    print(response.text)
def twilio_send_message(msg, to_phone, from_phone):

    #response = app.model.generate_content(msg)
    chat = None
    if to_phone in app.chat_dict:
        print('continuing conversation with ', to_phone)
        chat = app.chat_dict[to_phone]
    else:
        print(to_phone, ' not found, starting new chat')
        chat = app.model.start_chat(history=[])
        app.chat_dict[to_phone] = chat
    response = chat.send_message(msg)

    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

    message = client.messages.create(
        body=response.text,
        messaging_service_sid='MGe307ac508d8e4a350fb676f86cf017c4',
        #from_=from_phone,
        to=to_phone,
    )

    print(message)
    #Vonage
    #client = vonage.Client(key=VONAGE_API_KEY, secret=VONAGE_API_SECRET)
    #responseData = client.sms.send_message({
   #     "from": my_phone,
   #     "to": phone,
   #     "text": msg,
   # })

#    print(responseData)
#    if responseData["messages"][0]["status"] == "0":
#        print("Message sent successfully.")
#        return 204
#    else:
#        print(f"Message failed with error: {responseData['messages'][0]['error-text']}")
#        return 400

def receive_message(from_phone, message):
    user = User.query.filter_by(phone=from_phone).first()

    # Determine the right reply for this message
    if message in ['END', 'STOP', 'CANCEL', 'UNSUBSCRIBE']:
        response_string = "ending the conversation. Please text START to resume. Thank you."
        handle_consent(from_phone, consent=False)

    elif message == 'START':
        response_string = "Thanks for authorizing. Beginning conversation"
        handle_consent(from_phone)
        # Start a new thread to send the message asynchronously
        #thread = threading.Thread(target=send_message, args=(body, from_phone, to_phone,))
        #thread.start()
    
    elif (user and user.consent) or REQUIRE_CONSENT == False:
        chat = None
        # Check if the string starts with 'PROMPT'
        if message.startswith('PROMPT'):
            # Assign the remaining portion of the string (after 'PROMPT: ') to new_prompt
            #old_chat = app.chat_dict[from_phone] if from_phone in app.chat_dict else None
            #history = old_chat.history if old_chat else []
            new_prompt = message[len('PROMPT: '):]
            if new_prompt != '':
                model = reinitialize_model(new_prompt)
                chat = model.start_chat(history=[])
                app.chat_dict[from_phone] = chat
                response_string = "New chat session started with your prompt"
                return response_string
            else:
                return 'Error: PROMPT must be followed by the prompt string'
        
        elif from_phone in app.chat_dict:
            print('continuing conversation with ', from_phone)
            chat = app.chat_dict[from_phone]

        else:
            print(from_phone, ' not found, starting new chat')
            chat = app.model.start_chat(history=[])
            app.chat_dict[from_phone] = chat

        response = chat.send_message(message)
        response_string = response.text

        # Handle case where user is missing
        if user is None:
           user = handle_consent(from_phone)

        # Create a new message instance
        new_message = Message(
            prompt=message,
            response=response_string,
            user_id=user.user_id,
            phone=from_phone
        )
        db.session.add(new_message)

        # Commit the session to save the new message to the database
        db.session.commit()

    else:
        response_string = "Please text START to begin conversing with the AI agent"

    # Generate a random delay between 1 and 3 seconds
    delay = random.uniform(1, 3)
    
    # Wait for the random delay
    time.sleep(delay)
    return response_string

@app.route('/status')
def status():
    # Sample options for the dropdown
    options = ['Shy', 'Confident', 'Grouchy', 'Distracted']

    # Assuming system_instructions is retrieved from somewhere in your application
    system_instructions = 'Shy'

    return render_template('status.html', options=options, instructions=system_instructions)

# ----------------------------------------------------
# Chat api handling routes
@app.route('/chat', methods=['POST', 'GET'])
def chat():
    if request.method == 'POST':
        try:
            data = request.get_json()
            prompt = data.get('prompt')
            question = prompt
            print(question)
            # response = app.model.generate_content(question)
            response = app.chat.send_message(question)
            # Generate a random delay between 1 and 3 seconds
            delay = random.uniform(1, 3)
            
            # Wait for the random delay
            time.sleep(delay)
            if response.text:
                return response.text
            else:
                return "Bob:Sorry, but I think Gemini didn't want to answer that!"
        except Exception as e:
            print(e)
            return f"{e} Dob:Sorry, but Gemini didn't want to answer that!"

    return render_template('chat.html', **locals())

@app.route('/gpt-chat', methods=['POST'])
def gpt_chat():
    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({"error": "No message provided"}), 400

    payload = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": message}],
        "max_tokens": 150
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        error_message = str(e)
        if response is not None:
            try:
                error_details = response.json()
            except ValueError:
                error_details = response.text
        else:
            error_details = "No response from server"
        return jsonify({"error": "Failed to get response from OpenAI", "details": error_details, "message": error_message}), 500

    response_data = response.json()
    reply = response_data['choices'][0]['message']['content']

    return jsonify({"reply": reply})

@app.route('/gemini', methods=['POST'])
def gemini_chat():
    data = request.json
    message = data.get('message', '')

    if not message:
        return jsonify({"error": "No message provided"}), 400

    response = app.model.generate_content(message)
    print(response)
    return response.text

@app.route('/vonage/inbound-sms', methods=['GET', 'POST'])
def inbound_sms():
    print('request received')
    msg = ''
    phone = 0
    if request.is_json:
        #print(request.get_json())
        print(request.text)
        msg = request.text
        phone = request.msisdn
        my_phone = request.to
        print(phone)
        print(my_phone)
    else:
        data = dict(request.form) or dict(request.args)
        msg = data['text']
        phone = data['msisdn']
        my_phone = data['to']
        print(msg)
        print(phone)
        print(my_phone)

    if msg != '':
        response = app.model.generate_content(msg)
        send_message(response.text, phone, my_phone)

        print(response.text)
    return ('', 204)

@app.route("/webhooks/message-status", methods=["POST"])
def message_status():
    data = request.get_json()
    print('Message status: ' + data)
    return "200"

@app.route('/webhooks/delivery-receipt', methods=['GET', 'POST'])
def delivery_receipt():
    if request.is_json:
        pprint(request.get_json())
        pprint(request.status)
    else:
        data = dict(request.form) or dict(request.args)
        print('Status: ', data['status'])
        print('ISDM:', data['msisdn'])
        print("Error Code:", data['err-code'])

    return ('', 204)

#Twilio route
@app.route("/twilio/sms", methods=['GET', 'POST'])
def incoming_sms():
    """Send a dynamic reply to an incoming text message"""
    # Get the message the user sent our Twilio number
    body = request.values.get('Body', None)
    to_phone = request.values.get('To', None)
    from_phone = request.values.get('From', None)
    print(body)
    print(to_phone)
    print(from_phone)
    # Start our TwiML response
    resp = MessagingResponse()
    response_string = receive_message(from_phone, body)
    resp.message = response_string

    print(resp)
    return str(resp)

@app.route('/postack/sms', methods=['POST'])
def postack_sms():
    content = request.json
    nested_data = content.get('data')
    from_phone = nested_data.get('from')
    to_phone = nested_data.get('to')
    msg = nested_data.get('text')
    print(msg)
    print(to_phone)
    print(from_phone)
    response_text = receive_message(from_phone, msg)
    postack_send_message(response_text, from_phone, to_phone)
    print(response_text)
    return '200'
    

@app.route("/twilio/message-status", methods=['POST'])
def incoming_sms_status():
    message_sid = request.values.get('MessageSid', None)
    message_status = request.values.get('MessageStatus', None)
    print(message_status)
    logging.info('SID: {}, Status: {}'.format(message_sid, message_status))

    return ('', 204)

@app.route('/submit', methods=['POST'])
def submit():
    username = request.form.get('username')
    email = request.form.get('email')
    phone = request.form.get('phone')
    consent = request.form.get('permission') == 'on'
    print(username)
    print(email)
    print(phone)
    print(consent)

    if not username or not email or not phone or not consent:
        flash('Please fill out all fields', 'error')
        return redirect(url_for('signup'))
    user = User(username=username, email=email, phone=phone, consent=consent)
    db.session.add(user)
    db.session.commit()
    flash('Form submitted successfully!', 'success')
    return redirect('/')

@app.route('/thank-you')
def thank_you():
    return 'Thank you for signing up!'


if __name__ == '__main__':
    app.run(debug=True)

