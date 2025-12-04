from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Hej världen! Min första offentliga webbsida!</h1>"

if __name__ == "__main__":
    app.run()
