import eventlet
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
import threading
import os
import json

app = Flask(__name__)
socketio = SocketIO(app, message_queue='redis://localhost:6379/0', cors_allowed_origins="*")


@app.route('/')
def index():
    try:
        with open('output.log', 'r') as f:
            initial_content = f.read()
        # Escape the JSON to be safely embedded in the HTML
        initial_content_json = json.dumps(initial_content)
    except Exception as e:
        initial_content_json = json.dumps(f"Error reading log file: {str(e)}")
        app.logger.error(f"Failed to read log file: {str(e)}")
    
    return render_template('index.html', initial_content=initial_content_json)

def tail_f(filename, interval=1.0):
    try:
        with open(filename, 'r') as f:
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    eventlet.sleep(interval)
                    continue
                socketio.emit('log_update', json.dumps(line))
    except Exception as e:
        app.logger.error(f"Error in tail_f: {str(e)}")

@socketio.on('connect')
def handle_connect():
    app.logger.info('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    app.logger.info('Client disconnected')

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    threading.Thread(target=tail_f, args=('output.log', 1.0), daemon=True).start()
    eventlet.monkey_patch()
    socketio.run(app, host='0.0.0.0', port=8000)

    #eventlet.wsgi.server(eventlet.listen(('0.0.0.0', 8000)), app)
