# Repository

## Overview
Provides a backend API for repository analysis workflows; generates repository documentation from parsed source.

Project type: Unknown

Parsed surface: **9 files** · **26 functions** · **6 classes**

## Architecture / How It Works
- **api**: `routes/image_predict.py`, `routes/text_predict.py`, `routes/video_predict.py`
- **docs**: `readme.md`, `README.md`

## Project Structure
```text
root/
routes/
  - routes/image_predict.py
  - routes/text_predict.py
  - routes/video_predict.py
training-code/
  - training-code/image-train.py
  - training-code/text-train.py
  - training-code/video-train.py
main.py
```

## Key Components
- **`training-code/image-train.py`**: 
  - Symbols: `HuggingfaceImageDataset`, `BasicCNN`, `__init__()`, `__len__()`, `__getitem__()`
  - Imports: PIL, datasets, io, sklearn.metrics
- **`training-code/video-train.py`**: 
  - Symbols: `VideoDataset`, `CRNN`, `__init__()`, `__len__()`, `__getitem__()`
  - Imports: PIL, cv2, os, sklearn.metrics
- **`main.py`**: 
  - Symbols: `home()`, `image_page()`, `text_page()`
  - Imports: flask, os, routes.image_predict, routes.text_predict
- **`routes/video_predict.py`**: 
  - Symbols: `CRNN`, `extract_frames()`, `predict_video()`, `__init__()`
  - Imports: PIL, cv2, torch, torch.nn
- **`routes/image_predict.py`**: 
  - Symbols: `BasicCNN`, `predict_image()`, `__init__()`, `forward()`
  - Imports: PIL, os, torch, torch.nn
- **`training-code/text-train.py`**: 
  - Symbols: no detected symbols
  - Imports: pandas, sklearn.feature_extraction.text, sklearn.linear_model, sklearn.metrics
- **`routes/text_predict.py`**: 
  - Symbols: `predict()`
  - Imports: joblib, numpy
- **`readme.md`**: 
  - Symbols: no detected symbols
  - Imports: none
- **`README.md`**: 
  - Symbols: no detected symbols
  - Imports: none

## Technologies Used
- **py**: 7 file(s)
- **md**: 2 file(s)
- **Flask**
- **PyTorch**
- **Pandas**
- **NumPy**

## Usage
```bash
python main.py
```

## Notes / Limitations
- High coupling between `main.py` and `routes` modules.
- Use of multiple deep learning frameworks (PIL, torch, torchvision).
- Use of multiple machine learning libraries (sklearn, joblib).
- Lack of clear separation of concerns between training and prediction code.