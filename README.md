# CNN-based Chest X-ray Image Classification

This repository contains a deep learning project that classifies chest X-ray images into four categories: **Normal**, **Covid**, **Pneumonia**, and **Lung Opacity**. The project showcases several advanced techniques in image preprocessing, convolutional neural network (CNN) design, and model regularization—demonstrating a strong grasp of computer vision and deep learning fundamentals.

## Project Overview

- **Dataset:** The code loads chest X-ray images from a structured dataset (assumed to be in the Kaggle input path) and processes them for training.
- **Preprocessing:** Images are resized, normalized, and converted to grayscale, with labels one-hot encoded for multi-class classification.
- **Model Architecture:** A sequential CNN model is implemented with multiple convolutional layers, pooling layers, dropout, and dense layers.
- **Regularization Techniques:** Both dropout and L1/L2 regularization are applied to combat overfitting.
- **Data Augmentation (Optional):** The code includes a commented-out section using `ImageDataGenerator` to apply transformations and a custom Gaussian noise function to enhance training data robustness.
- **Training and Evaluation:** The dataset is split into training, validation, and test sets. The model is trained and then evaluated on a separate test set, with loss and accuracy metrics reported.

## Detailed Breakdown

### Data Loading and Preprocessing

- **Image Handling:** 
  - Uses Python’s PIL library to open and convert images to NumPy arrays.
  - Images are reshaped to a fixed size of 299x299 with one channel to suit the CNN input requirements.
- **Label Encoding:** 
  - Labels are generated based on directory names and one-hot encoded using Keras’s `to_categorical`.
- **Dataset Splitting:**
  - The dataset is divided into training, validation, and test sets using `train_test_split` from scikit-learn.

### Model Architecture

- **Convolutional Layers:**
  - Multiple `Conv2D` layers extract high-level features from input images.
  - The first convolutional layer employs L1 and L2 regularization (`regularizers.l1_l2`) to reduce overfitting.
- **Pooling and Dropout:**
  - `MaxPooling2D` layers reduce spatial dimensions, lowering computational cost.
  - `Dropout` layers randomly deactivate neurons during training, enhancing model generalization.
- **Fully Connected Layers:**
  - After flattening the feature maps, dense layers are applied, culminating in a softmax output layer for multi-class classification.

### Training and Evaluation

- **Compilation:**
  - The model is compiled with the Adam optimizer and categorical crossentropy loss.
- **Training Process:**
  - The model is initially trained for 5 epochs and then further trained for 75 epochs to refine its performance.
- **Evaluation:**
  - Final model performance is assessed on a separate test set with metrics for loss and accuracy printed out.

### Data Augmentation (Optional)

- **Techniques Included:**
  - Featurewise centering and standard normalization.
  - Random height shifts, zooms, and horizontal flips to augment data variability.
  - A custom preprocessing function adds Gaussian noise to images, enhancing robustness.
- **Usage:**
  - Although commented out in this version, the data augmentation block can be activated to expand the training dataset artificially.

## Technologies and Techniques

- **Python & NumPy:** For efficient data manipulation and image processing.
- **PIL (Pillow):** To handle image file I/O and conversion.
- **TensorFlow/Keras:** For building, training, and evaluating deep learning models.
- **scikit-learn:** For dataset splitting and performance evaluation.
- **Regularization & Augmentation:** Techniques like dropout, L1/L2 regularization, and data augmentation improve model robustness and prevent overfitting.

