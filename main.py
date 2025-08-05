from flask import Flask, render_template, request
from routes.text_predict import predict  # make sure this points to the updated predict
from routes.image_predict import predict_image  # make sure this points to the updated predict
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/image')
def image_page():
    return render_template('img_index.html')  

@app.route('/text')
def text_page():
    return render_template('index.html')   

@app.route('/video')
def video_page():
    return render_template('video.html')  

@app.route("/predict", methods=["POST"])
def predict_route():
    text = request.form.get("text", "")
    label, score = predict(text)
    return render_template("result.html", text=text, label=label, score=score)

@app.route('/image-predict', methods=['GET', 'POST'])
def image_route():
    if request.method == 'POST':
        uploaded_file = request.files['image']
        if uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            uploaded_file.save(filepath)

            label, confidence = predict_image(filepath)
            return render_template('result.html', label=label, score=confidence, image_path=filepath)
    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)
