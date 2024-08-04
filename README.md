# SpamWaster-SMS

SpamWaster-telegram is a Python application that allows you to use a Telegram app to respond to messages using an LLM backed chat bot

## Features

- Automatically respond to spam messages using LLMs (Gemini, Claude, ChatGPT)
- Handle Telegram messages
- Easy setup and configuration with environment variables

## Prerequisites

- Python 3.6+
- Flask
- API accounts with Groq, Telegram, and db credentials

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
    DB_USER=username
    DB_PASSWORD=password
    DB_HOST=dbhost
    DB_NAME=dbname
    OPENAI_API_KEY=needed to use OpenAI models
    OPENAI_API_URL=eeded to use OpenAI models
    TELEGRAPH_APP_ID=contact Telegram to get this
    TELEGRAPH_API_HASH=contact Telegram to get this
    TELEGRAPH_PHONE=your telegram phone number
    HUGGING_FACE_ACCESS_TOKEN=contact Hugging Face to use this for model inference
    USE_HISTORY=True <-- use this variable to use stored messages as history context
    USE_DELAY=False <-- use this variable to add a random delay to responses
    AI_MODEL=name of the AI model
    GROQ_API_KEY=contact Groq to acquire a key and use models through their api
## Usage

**Run the Python application:**

    ```bash
    python spamwaster-telegram.py
    ```


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
