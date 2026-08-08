# Repository

## Overview
A Flask-based application providing prediction interfaces for image, video, and text data, supported by utilizing trained machine learning models.

## Architecture / How It Works
The application is divided into training and prediction layers:
- **Training**: Located in `training-code/`, these modules define datasets and model architectures (CNN, CRNN) and handle the training process.
- **Prediction**: Located in `routes/`, these modules load models to perform inference on provided inputs.
- **Routing**: `main.py` serves as the entry point, mapping web requests to the corresponding prediction handlers.

## Project Structure
```text
.
├── main.py
├── routes/
│   ├── image_predict.py
│   ├── text_predict.py
│   └── video_predict.py
└── training-code/
    ├── image-train.py
    ├── text-train.py
    └── video-train.py
```

## Key Components

### Image Processing
- **`training-code/image-train.py`**: Implements `HuggingfaceImageDataset` for data loading and `BasicCNN` for model architecture.
- **`routes/image_predict.py`**: Implements `BasicCNN` and the `predict_image` function for inference.

### Video Processing
- **`training-code/video-train.py`**: Implements `VideoDataset` and the `CRNN` model architecture.
- **`routes/video_predict.py`**: Implements `CRNN` and utility functions `extract_frames` and `predict_video`.

### Text Processing
- **`training-code/text-train.py`**: Handles text feature extraction and model training using `sklearn`.
- **`routes/text_predict.py`**: Implements the `predict` function using `joblib` for model loading.

### Application Entry
- **`main.py`**: Contains Flask route handlers including `home`, `image_page`, and `text_page`.

## Technologies Used
- **Framework**: Flask
- **Deep Learning**: PyTorch, Torchvision
- **Data Science**: Scikit-learn, Pandas, NumPy
- **Image/Video Processing**: OpenCV (cv2), PIL
- **Utilities**: Joblib, Werkzeug

## Usage
```bash
python main.py
```

## Notes / Limitations
- Model definitions (e.g., `BasicCNN`, `CRNN`) are duplicated across both training and prediction modules.
- Text training and prediction modules do not share a common class-based model definition.