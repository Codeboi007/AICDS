### **AICDS - AI Content Detection System**

AICDS is a **deep learning-powered system** that detects whether content—images or videos—is **AI-generated or real**. Using tailored CNN, CRNN, and pretrained ResNet-based architectures, it provides reliable classification results through a Flask web interface.

-----

### 💡 **Features**

  - Classifies **images** and **videos** into **AI-generated** or **Real**.
  - **Models**:
      - **CNN** for image classification.
      - **CRNN** (CNN + GRU) for video classification.
      - **ResNet18 + GRU** (Transfer learning) for improved video accuracy.
  - Detailed performance metrics including confusion matrix and classification report.
  - **Flask-based web interface** for real-time prediction.
  - Modular codebase, organized for easy training and deployment.

-----

### 📂 **Project Structure**

```bash
AICDS/
├── backend/
│ ├── image-train.py
│ ├── video-train.py
│ └── video-train-resnet.py
├── models/
│ ├── cnn_image_model.pth
│ ├── video_crnn_model.pth
│ └── video_crnn_resnet_model.pth
├── routes/
│ ├── image_predict.py
│ └── video_predict.py
├── templates/
│ └── index.html
├── training-data/
│ ├── image-data/
│ └── video-data/
├── main.py
└── requirements.txt
```

-----

### 🛠️ **Setup & Usage**

#### Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

#### Step 2: Train Models

Train any of the models depending on your content:

```bash
# Image model (CNN)
python backend/image-train.py

# Video model (CRNN)
python backend/video-train.py

# Video model (ResNet18 + GRU)
python backend/video-train-resnet.py
```

#### Step 3: Run Flask App

```bash
python main.py
```

Then, go to `http://localhost:5000` in your browser.

-----

### 🧪 **Dataset**

You can download the training data from the following links:

  - **Image Training Data**: [\[Link to Image Dataset\]](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images)
  - **Video Training Data**: [\[Link to Video Dataset\]](https://www.kaggle.com/datasets/kanzeus/realai-video-dataset)
  - **Text Training Data**: [\[Link to Text Dataset\]](https://www.kaggle.com/datasets/sunilthite/llm-detect-ai-generated-text-dataset)



-----

### 🔮 **Future Improvements**

  - Add real-time webcam video classification.
  - Expand to include text and audio-based AI detection.
  - Build an online dashboard with logs and statistics.
  - Optimize models for mobile and embedded systems.

