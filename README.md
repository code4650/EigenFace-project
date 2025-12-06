### Project Overview

This project uses traditional Machine Learning algorithm, Principal Component Analysis, PCA to classify human faces.  The goal is to develop a training and testing pipeline of PCA+SVM recognition on face images that includes augmentation and usage of several face detectors (Haar cascade, dlib HOG, custom HOG+SVM). The python file named EigenFace_webcam.py is a webcam demo for live recognition.
### Key Features

Training: PCA (Eigenfaces) + SVM pipeline built with scikit-learn inside EigenFace.ipynb.
Data Augmentation: Uses torchvision.transforms for randomized augmentation (rotation, flips, resize, normalization).
Face Detection: Supports Haar Cascade (OpenCV), dlib HOG detector, and a custom HOG+SVM face detector (joblib-loaded model).
Evaluation & Visualization: Accuracy metrics, confusion matrix, and PCA-component vs accuracy plots.
Webcam Demo: EigenFace_webcam.py — loads recognition_model.pkl and class_mapping.pkl to run real-time recognition.
### Files

EigenFace.ipynb: Jupyter notebook. Contains data loading, augmentation, training, PCA analysis, evaluation, and several detection variants.
EigenFace_webcam.py: Lightweight script to run webcam face recognition (uses Haar cascade).
Model outputs: recognition_model.pkl, class_mapping.pkl (saved by the notebook after training).
### Python / Platform

Recommended Python: 3.9.13 (this repo's tested compatibility).
Platform: Windows is used in examples; webcam demo works on any OS with camera access.
### Minimal Requirements

See the bundled requirements.txt (pinned to versions compatible with Python 3.9.13).
Core packages: numpy, opencv-python, joblib, torch, torchvision, scikit-learn, dlib, matplotlib, seaborn, scikit-image, Pillow, ipython.
Setup (PowerShell)

Upgrade pip and install dependencies:
Recommended: install PyTorch with the wheel matching your system first (CPU-only example):
### Run / Usage

Train and evaluate:
Open EigenFace.ipynb in Jupyter and run cells. Update the dataset path variables (currently set to Kaggle paths like /kaggle/input/...) to your local dataset directory.
Notebook training saves recognition_model.pkl and class_mapping.pkl.
Webcam demo:
Notes:
Press q in the webcam window to exit.
Ensure recognition_model.pkl and class_mapping.pkl are present in the same directory when running the webcam script.
### Dataset

The notebook expects a folder structure like:
DATA_ROOT/<person_name>/*.jpg|png
Update data_path variables in the notebook to point to your local dataset. The code filters common image extensions and skips non-image files.
### Outputs

recognition_model.pkl: Saved scikit-learn pipeline (StandardScaler → PCA → SVM).
class_mapping.pkl: Dictionary mapping integer labels → person names.
### Troubleshooting & Notes

dlib: Installing dlib on Windows often requires Visual Studio Build Tools and CMake. If pip install dlib fails, use:
Conda: conda install -c conda-forge dlib
Or install Visual Studio Build Tools + CMake then retry pip.
PyTorch (torch): Choose the correct wheel for CPU vs CUDA at https://pytorch.org. Installing via the site’s recommended command is safest.
OpenCV (cv2): haarcascade_frontalface_default.xml is loaded via cv2.data.haarcascades; no extra download required.
Large PCA components / memory: PCA with many components and large augmented datasets uses significant RAM — reduce num_augmentations or PCA components if you run out of memory.
Paths: Notebook uses absolute Kaggle paths (e.g., /kaggle/input/...). Replace with your local relative or absolute paths before running.
