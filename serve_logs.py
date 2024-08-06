from flask import Flask, Response, render_template_string

app = Flask(__name__)

@app.route('/')
def index():
    return render_template_string("""
    <!doctype html>
    <html>
      <head>
        <title>Log Stream</title>
      </head>
      <body>
        <h1>Log Stream</h1>
        <div id="log"></div>
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
    """)

@app.route('/stream')
def stream():
    def generate():
        with open('output.log', 'r') as f:
            f.seek(0, 2)  # Move the cursor to the end of the file
            while True:
                line = f.readline()
                if not line:
                    time.sleep(1)
                    continue
                yield f"data: {line}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
