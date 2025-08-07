from flask import Flask, render_template, request
from routes.text_predict import predict  
from routes.image_predict import predict_image  
from routes.video_predict import predict_video
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
    return render_template('video_index.html')  

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

@app.route("/video-predict", methods=["POST"])
def video_route():
    if 'video' not in request.files:
        return render_template("result.html", text="No file", label="Error", score=0)
    
    file = request.files['video']
    if file.filename == "":
        return render_template("result.html", text="No file", label="Error", score=0)

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    label, confidence = predict_video(filepath)
    return render_template("result.html", text=file.filename, label=label, score=confidence)

if __name__ == "__main__":
    app.run(debug=True)
