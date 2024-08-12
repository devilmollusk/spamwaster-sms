from flask import Flask, Response, render_template_string
import os
import time
import select

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
            f.seek(0, os.SEEK_END)  # Move the cursor to the end of the file
            while True:
                # Use select to wait for data to be available on the file descriptor
                rlist, _, _ = select.select([f], [], [], 1)
                if rlist:
                    line = f.readline()
                    if line:
                        yield f"data: {line}\n\n"
                else:
                    # No new data, continue to the next iteration
                    continue
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
