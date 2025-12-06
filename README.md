Here is a complete, ready-to-write `README.md` for your project. Copy the block below into `README.md` at the repository root (or I can create it for you).

```markdown
# EigenFace-project

## Project Overview
EigenFace-project implements classical face recognition using Principal Component Analysis (Eigenfaces) combined with a Support Vector Machine (SVM) classifier. The repository contains a Jupyter notebook that trains and evaluates a PCA+SVM recognition pipeline with data augmentation and several face detectors (OpenCV Haar Cascade, dlib HOG, and a custom HOG+SVM). A simple webcam demo script (`EigenFace_webcam.py`) runs live recognition using a trained model.

## Key Features
- **PCA + SVM Training**: Implements an Eigenfaces pipeline (PCA for dimensionality reduction + linear SVM) using `scikit-learn`.
- **Data Augmentation**: Uses `torchvision.transforms` for augmentations (random rotation, horizontal flip, resize, normalization).
- **Face Detection Options**:
  - OpenCV Haar Cascade (`cv2`).
  - dlib HOG detector.
  - Custom HOG+SVM detector (loadable via `joblib`).
- **Evaluation & Visualization**: Accuracy metrics, confusion matrix plotting, and PCA components vs accuracy exploration.
- **Webcam Demo**: `EigenFace_webcam.py` for real-time recognition; uses `recognition_model.pkl` and `class_mapping.pkl`.

## Repository Structure
- `EigenFace.ipynb` — Jupyter notebook for dataset loading, augmentation, PCA, training, evaluation, and visualization.
- `EigenFace_webcam.py` — Lightweight script for webcam recognition (Haar cascade demo).
- `requirements.txt` — Project dependencies (recommended to pin versions for reproducibility).
- Model artifacts (created by the notebook):
  - `recognition_model.pkl` — Saved scikit-learn pipeline: `StandardScaler → PCA → SVM`.
  - `class_mapping.pkl` — Mapping from numeric class index to person name.

## Python / Platform
- **Recommended Python**: `3.9.13`
- **Platform**: Examples and instructions assume Windows (PowerShell). The webcam demo works on other OSes as long as a camera and appropriate drivers are available.

## Dependencies
Required packages include (see `requirements.txt` in repo for pinned versions):
- numpy
- opencv-python
- joblib
- torch
- torchvision
- scikit-learn
- dlib
- matplotlib
- seaborn
- scikit-image
- Pillow
- ipython

## Installation (PowerShell)
1. Upgrade pip:
```powershell
python -m pip install --upgrade pip
```
2. Recommended: install PyTorch first (choose CPU or CUDA wheel at https://pytorch.org). CPU example:
```powershell
pip install torch==1.13.1+cpu torchvision==0.14.1+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
```
3. Install the rest of the dependencies from `requirements.txt`:
```powershell
pip install -r requirements.txt --no-deps
```
Notes:
- Installing `dlib` on Windows commonly requires Visual Studio Build Tools and CMake. If `pip install dlib` fails, use:
  - Conda: `conda install -c conda-forge dlib`  
  - Or install Visual Studio Build Tools + CMake then retry `pip install dlib`.
- For GPU support, pick the correct PyTorch command from the official site.

## Usage

### Training & Evaluation (Notebook)
1. Open `EigenFace.ipynb` in JupyterLab / Jupyter Notebook.
2. Update dataset paths in the notebook (`data_path` variables). The notebook examples use Kaggle-style paths (e.g., `/kaggle/input/...`); replace these with your local dataset path.
3. Run cells to:
   - Load images and augment them.
   - Fit the PCA + SVM pipeline.
   - Evaluate and visualize results.
4. Training saves `recognition_model.pkl` and `class_mapping.pkl` to the notebook directory.

### Webcam Demo
Ensure `recognition_model.pkl` and `class_mapping.pkl` are in the same folder as `EigenFace_webcam.py`, then run:
```powershell
python EigenFace_webcam.py
```
- Press `q` in the webcam window to quit.

## Dataset Format
Expected dataset folder layout:
```
DATA_ROOT/
  Person_A/
    img1.jpg
    img2.png
    ...
  Person_B/
    img1.jpg
    ...
```
- The notebook filters common image extensions and skips non-image files. Update the `data_path` variable to point to `DATA_ROOT`.

## Outputs
- `recognition_model.pkl`: Serialized scikit-learn pipeline (`StandardScaler → PCA → SVM`) used for inference.
- `class_mapping.pkl`: Dictionary mapping numeric labels to person names.

## Troubleshooting & Tips
- dlib: On Windows install Visual Studio Build Tools and CMake before pip if you need to build from source; otherwise use conda-forge wheels.
- PyTorch: Use the official installer for the matching CUDA/CPU wheel.
- OpenCV cascades: Haar cascade files are accessible via `cv2.data.haarcascades` (no manual download required).
- Memory: Large PCA component choices and high augmentation counts increase memory usage — lower `num_augmentations` or PCA components if you run out of RAM.
- Paths: Replace absolute Kaggle paths in the notebook with local paths before running locally.



