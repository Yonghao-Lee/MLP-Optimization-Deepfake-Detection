# Spatial Classification & Deepfake Detection

This repository contains the implementation of **Machine Learning** models for two distinct tasks:
1.  **Spatial Classification:** A Multi-Layer Perceptron (MLP) to predict the country of a city based on its coordinates (Latitude/Longitude).
2.  **Deepfake Detection:** A Convolutional Neural Network (ResNet18) to classify images as real or fake (AI-generated).

##  Quick Start
1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run Spatial Classification (MLP):**
    ```bash
    jupyter notebook notebooks/spatial_classification.ipynb
    ```
3.  **Run Deepfake Detection (CNN):**
    ```bash
    jupyter notebook notebooks/deepfake_detection.ipynb
    ```

##  Models & Results

### 1. MLP for Spatial Classification
* **Best Config:** Depth 6, Width 16.
* **Accuracy:** ~95.5% (Test).
* **Key Feature:** Implements **Residual Connections** (ResNet-style) to prevent vanishing gradients in deeper networks.

### 2. CNN for Deepfake Detection
* **Model:** ResNet18 (Fine-Tuned).
* **Accuracy:** 85.0%.
* **Comparison:** Outperformed Linear Probing (73%) and Training from Scratch (57%).
* **Insight:** Transfer learning allowed the model to detect subtle GAN artifacts (e.g., anatomical inconsistencies) that models trained from scratch missed.

##  Performance Analysis
**Observation:** During the MLP training, the **Test/Validation Accuracy** was observed to be consistently higher than the **Training Accuracy** (or Training Loss was higher than Test Loss).

**Explanation:** This anomaly is a known effect of **Batch Normalization (BN)**.
* **During Training:** BN layers normalize using the *current mini-batch* statistics, which introduces noise and variability, effectively acting as a regularizer.
* **During Testing:** BN layers use the *running average* (population) statistics, which are stable and precise.
Consequently, the model often performs "better" on the validation set (using stable stats) than on the training set (using noisy batch stats), especially in early epochs or with smaller batch sizes.
