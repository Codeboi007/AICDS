# Overview
This repository contains a backend API for handling predictions on images, videos, and text. It includes training code for each media type and prediction routes for serving predictions.

# Architecture / How It Works
- **API Routes**: The `routes` directory contains modules for handling prediction requests for images, videos, and text.
- **Training Code**: The `training-code` directory includes scripts for training models on images, videos, and text.

# Project Structure
```
root/
├── main.py
├── training-code/
│   ├── image-train.py
│   ├── video-train.py
│   └── text-train.py
└── routes/
    ├── image_predict.py
    ├── text_predict.py
    └── video_predict.py
```

# Key Components
- **`training-code/image-train.py`**: Contains `HuggingfaceImageDataset` and `BasicCNN` classes for image training.
- **`training-code/video-train.py`**: Contains `VideoDataset` and `CRNN` classes for video training.
- **`main.py`**: Defines routes for the home page and pages for image and text predictions.
- **`routes/video_predict.py`**: Implements video prediction using the `CRNN` class.
- **`routes/image_predict.py`**: Implements image prediction using the `BasicCNN` class.
- **`training-code/text-train.py`**: Contains training code for text data.
- **`routes/text_predict.py`**: Implements text prediction.

# Technologies Used
- **Flask**: For creating the web API.
- **PyTorch**: For building and training neural networks.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **OpenCV (cv2)**: For video processing.
- **PIL**: For image processing.
- **Scikit-learn**: For machine learning utilities and metrics.

# Usage
```bash
python main.py
```

# Notes / Limitations
- **Lack of separation between training and prediction code** might lead to maintenance issues.
- **High dependency on specific libraries** (e.g., torch, sklearn) could pose risks if these libraries are deprecated or have breaking changes.
- **No clear separation of concerns in `main.py`**, which handles routing and prediction logic.