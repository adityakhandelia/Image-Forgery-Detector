- Image Forgery Detection using VGG16 (CASIA v2)
- Overview

This project focuses on detecting tampered (forged) vs authentic images using a deep learning approach based on the VGG16 Convolutional Neural Network.

The model is trained on the CASIA v2 dataset, a widely used benchmark for image forgery detection. The system learns discriminative features to classify whether an image has been manipulated.

- Features
Binary classification: Authentic vs Tampered
Transfer learning using pretrained VGG16
Custom dataset loader for CASIA v2 structure
Train / Validation / Test split
Evaluation using multiple metrics:
Accuracy
Precision
Recall
F1 Score
Visualization:
Loss vs Epoch graph
Accuracy vs Epoch graph
Confusion Matrix
Robust handling of corrupted/non-image files
- Dataset

The project uses the CASIA v2 dataset, organized as:

CASIA2/
 ├── Au  (Authentic images)
 ├── Tp  (Tampered images)
 └── CASIA 2 Groundtruth (optional masks)
Authentic images → Label 0
Tampered images → Label 1

Dataset is split into:

70% Training
15% Validation
15% Testing
- Model Architecture
Base model: VGG16 (Pretrained on ImageNet)
Modified final layer for binary classification
Optional freezing of feature extractor layers
- Tech Stack
Python
PyTorch
Torchvision
Scikit-learn
Matplotlib & Seaborn
  - Training Pipeline
Load dataset (Au & Tp folders)
Filter valid image files
Apply preprocessing & normalization
Split into train/val/test
Train VGG16 model
Validate per epoch
Evaluate on test set
Generate performance metrics and graphs
- Results

Typical performance on CASIA v2:

Metric	Score
Accuracy	~75–88%
Precision	~0.80
Recall	~0.78
F1 Score	~0.79

Note: Performance can be improved using patch-based training and preprocessing techniques.

- Visualizations
Training vs Validation Accuracy
Training vs Validation Loss
Confusion Matrix
