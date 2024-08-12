from flask import Flask, Response, render_template_string
import time

app = Flask(__name__)

@app.route('/')
def index():
    # Read the entire log file content for initial load
    with open('output.log', 'r') as f:
        log_content = f.read()

    return render_template_string("""
    <!doctype html>
    <html>
      <head>
        <title>Log Stream</title>
      </head>
      <body>
        <h1>Log Stream</h1>
        <div id="log">{{ log_content|safe }}</div>
        <script type="text/javascript">
          const eventSource = new EventSource("/stream");
          eventSource.onmessage = function(event) {
            const log = document.getElementById("log");
            const newElement = document.createElement("div");
            newElement.innerHTML = event.data.replace(/\\n/g, '<br>');
            log.appendChild(newElement);
          };
        </script>
      </body>
    </html>
    """, log_content=log_content.replace('\n', '<br>'))

@app.route('/stream')
def stream():
    def generate():
        with open('output.log', 'r') as f:
            f.seek(0, 2)  # Move the cursor to the end of the file
            while True:
                line = f.readline()
                if not line:
                    try:
                        time.sleep(1)  # Sleep for 1 second
                    except Exception as e:
                        print(f"Error during sleep: {e}")
                    continue
                yield f"data: {line}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
