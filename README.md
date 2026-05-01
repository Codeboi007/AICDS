# Overview
This repository contains a backend API for handling predictions on images, videos, and text. It includes training code for image and video datasets and prediction routes for each media type.

# Architecture / How It Works
- **API Routes**: The `routes` directory contains prediction routes for images, videos, and text.
- **Training Code**: The `training-code` directory includes training scripts for images, videos, and text.

# Project Structure
```
root/
routes/
  - image_predict.py
  - text_predict.py
  - video_predict.py
training-code/
  - image-train.py
  - text-train.py
  - video-train.py
main.py
README.md
readme.md
```

# Key Components
- **`training-code/image-train.py`**: Contains `HuggingfaceImageDataset` and `BasicCNN` classes with methods `__init__()`, `__len__()`, and `__getitem__()`.
- **`training-code/video-train.py`**: Contains `VideoDataset` and `CRNN` classes with methods `__init__()`, `__len__()`, and `__getitem__()`.
- **`main.py`**: Defines routes `home()`, `image_page()`, and `text_page()`.
- **`routes/video_predict.py`**: Includes `CRNN` class and functions `extract_frames()` and `predict_video()`.
- **`routes/image_predict.py`**: Includes `BasicCNN` class and functions `predict_image()` and `forward()`.
- **`training-code/text-train.py`**: Contains training code for text data.
- **`routes/text_predict.py`**: Includes function `predict()` for text predictions.

# Technologies Used
- **Flask**: For web framework.
- **PyTorch**: For deep learning models.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical operations.
- **PIL**: For image processing.
- **OpenCV (cv2)**: For video processing.
- **Scikit-learn (sklearn)**: For machine learning utilities.

# Usage
```bash
python main.py
```

# Notes / Limitations
- Multiple README files (`README.md` and `readme.md`) may cause confusion.
- No clear separation between training and prediction code, which could lead to maintenance issues.
- High dependency on external libraries (e.g., torch, sklearn) which could pose versioning and compatibility risks.