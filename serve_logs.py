from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import threading
import time
import os

app = Flask(__name__)
socketio = SocketIO(app)

@app.route('/')
def index():
    with open('output.log', 'r') as f:
        initial_content = f.read().replace('\n', '<br>')  # Preserve line breaks
    return render_template('index.html', initial_content=initial_content)

def tail_f(filename, interval=1.0):
    """
    Mimics the behavior of tail -f to monitor the file for changes.
    """
    with open(filename, 'r') as f:
        f.seek(0, os.SEEK_END)  # Start at the end of the file
        while True:
            line = f.readline()
            if not line:
                time.sleep(interval)
                continue
            socketio.emit('log_update', line.replace('\n', '<br>'))

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == "__main__":
    # Start a background thread to monitor the log file
    threading.Thread(target=tail_f, args=('output.log', 1.0), daemon=True).start()
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)
