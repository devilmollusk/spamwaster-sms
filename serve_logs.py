from flask import Flask, Response, render_template
import time

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    def generate():
        with open('output.log', 'r') as f:
            f.seek(0, 2)  # Move the cursor to the end of the file
            while True:
                line = f.readline()
                if not line:
                    time.sleep(1)  # Wait for new data
                    continue
                yield f"data: {line}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
