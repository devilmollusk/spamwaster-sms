# SpamWaster-SMS

SpamWaster-SMS is a Python Flask application that sets up phone numbers and WhatsApp accounts to chat with spammers using Large Language Models like Gemini, Claude, and ChatGPT. The app has dedicated paths to handle events from chat services like Twilio, Vonage, and Postack.

## Features

- Automatically respond to spam messages using LLMs (Gemini, Claude, ChatGPT)
- Handle SMS and WhatsApp messages from various services
- Easy setup and configuration with environment variables

## Prerequisites

- Python 3.8+
- Flask
- API accounts with Twilio, Vonage, and Postack

## Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/your_username/spamwaster-sms.git
    cd spamwaster-sms
    ```

2. **Create a virtual environment and activate it:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Create a `.env` file in the root directory and add your API keys and secrets:**

    ```plaintext
    TWILIO_ACCOUNT_SID=your_twilio_account_sid
    TWILIO_AUTH_TOKEN=your_twilio_auth_token
    TWILIO_PHONE_NUMBER=your_twilio_phone_number

    VONAGE_API_KEY=your_vonage_api_key
    VONAGE_API_SECRET=your_vonage_api_secret
    VONAGE_PHONE_NUMBER=your_vonage_phone_number

    POSTACK_API_KEY=your_postack_api_key
    POSTACK_API_SECRET=your_postack_api_secret
    POSTACK_PHONE_NUMBER=your_postack_phone_number

    GEMINI_API_KEY=your_gemini_api_key
    CLAUDE_API_KEY=your_claude_api_key
    CHATGPT_API_KEY=your_chatgpt_api_key
    ```

## Usage

1. **Run the Flask application:**

    ```bash
    flask run
    ```

2. **Expose your local server to the internet using a tool like ngrok (for local development):**

    ```bash
    ngrok http 5000
    ```

3. **Configure your Twilio, Vonage, and Postack webhooks to point to your public ngrok URL. For example:**

    ```
    https://your-ngrok-url.ngrok.io/twilio
    https://your-ngrok-url.ngrok.io/vonage
    https://your-ngrok-url.ngrok.io/postack
    ```

## Flask Routes

- `/twilio`: Handles incoming messages from Twilio.
- `/vonage`: Handles incoming messages from Vonage.
- `/postack`: Handles incoming messages from Postack.

## Contributing

1. **Fork the repository**
2. **Create a new branch (`git checkout -b feature-branch`)**
3. **Commit your changes (`git commit -am 'Add some feature'`)**
4. **Push to the branch (`git push origin feature-branch`)**
5. **Create a new Pull Request**

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Inspired by the potential of Large Language Models to interact with spammers.
- Thanks to the developers of Flask and other open-source libraries used in this project.
