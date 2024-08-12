from flask import Flask, Response, render_template
import time

app = Flask(__name__)

@app.route('/')
def index():
    # Read the entire log file content to serve on initial load
    with open('output.log', 'r') as f:
        initial_content = f.read().replace('\n', '<br>')  # Preserve line breaks
    return render_template('index.html', initial_content=initial_content)

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
                yield f"data: {line.replace('\n', '<br>')}\n\n"  # Preserve line breaks
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
