# Football Play Analysis Project

This project provides an automated pipeline for extracting individual American football plays from game film and classifying them into offensive categories (Run vs. Pass). It combines traditional computer vision techniques for motion detection with modern deep learning for video sequence classification.

## Core Components

### 1. Snap Detection (`detectSnap.ipynb`)
This script serves as the data preprocessing engine by identifying the start of a play within a continuous video file.
* **Motion Analysis**: Uses **OpenCV** and the Farneback dense optical flow algorithm to monitor pixel-level movement.
* **Heuristic Triggering**: Identifies a "snap" when the running average of motion magnitude exceeds a predefined threshold.
* **Automated Clipping**: Utilizes **MoviePy** to extract subclips, capturing 1 second prior to the snap and 5 seconds of follow-through.
* **Deduplication**: Implements a time-buffer to ensure multiple triggers during the same play do not result in redundant clips.

### 2. Model Training (`training.ipynb`)
This notebook focuses on fine-tuning a state-of-the-art video classifier.
* **Architecture**: Employs the **TimeSformer** (Time-Space Transformer) architecture from the Hugging Face `transformers` library.
* **Transfer Learning**: Loads a model pre-trained on the Kinetics-400 dataset and freezes the transformer backbone to train a custom classification head.
* **Data Pipeline**: Features a custom `VideoDataset` and a complex `collate_fn` to handle the padding and 5D tensor reshaping $(Batch, Channels, Frames, Height, Width)$ required for Transformer-based video input.
* **Optimization**: Uses Cross-Entropy Loss and the Adam optimizer to distinguish between Run (0) and Pass (1) labels.

### 3. Testing & Inference (`testing.ipynb`)
A dedicated environment for validating model accuracy and experimenting with alternative architectures.
* **Alternative Architecture**: Contains a custom **ResNet-GRU** hybrid model that uses ResNet-18 for spatial feature extraction and a GRU for temporal sequence modeling.
* **Evaluation Metrics**: Provides logic to calculate Accuracy, Precision, Recall, and F1-Score to assess how well the model generalizes to new game footage.

## Project Requirements
* **Computer Vision**: `opencv-python`, `moviepy`, `imagehash`, `Pillow`
* **Deep Learning**: `torch`, `torchvision`, `transformers`
* **Data Science**: `numpy`, `scikit-learn`

## Performance Summary
Based on recent evaluation reports, the classification models have achieved the following benchmarks:

| Metric | Best Performance |
| :--- | :--- |
| **Accuracy** | 77.8% |
| **Precision (Positive)** | 0.75 |
| **Recall (Positive)** | 1.00 |
| **F1-Score (Positive)** | 0.857 |

---
