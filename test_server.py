print("Script started...")

from flask import Flask
print("Flask imported OK")

app = Flask(__name__)

@app.route('/')
def home():
    return '<h1>Server is working!</h1>'

print("Starting server on port 8080...")
app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
print("This line should never print while server is running")