from flask import Flask, render_template, request
from routes.text_predict import predict  # make sure this points to the updated predict

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict_route():
    text = request.form.get("text", "")
    label, score = predict(text)
    return render_template("result.html", text=text, label=label, score=score)

if __name__ == "__main__":
    app.run(debug=True)
