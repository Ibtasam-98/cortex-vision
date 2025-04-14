# Cortex Vision (ML Backend)

This repository contains the Python-based machine learning backend for the Cortex Vision mobile application. This backend is responsible for receiving eye images, processing them using machine learning algorithms, and predicting the likelihood of cataracts. It exposes a Flask API for communication with the mobile application.

## Overview

The Cortex Vision mobile application (available in a separate repository) allows users to upload images of their eyes. This backend processes these images to detect features relevant to cataract diagnosis and utilizes trained machine learning models to generate a prediction. The communication between the mobile app and this backend is done via a RESTful Flask API.

## Repository Structure

## Key Components

* **`dataset/` (Private):** This directory contains the image dataset used to train the machine learning models. Due to data privacy and size considerations, this directory is not publicly accessible.
  
* **`evaluation.py`:** This script is used to evaluate the performance of the trained machine learning models on a held-out test set. It typically calculates metrics like accuracy, precision, recall, and F1-score.
  
* **`feature_extraction.py`:** This module implements functions to extract relevant features from the input eye images. These features are then used by the machine learning models for prediction. Common techniques might include analyzing texture, color, or specific image patterns.
  
* **`main.py`:** This is the core of the backend API. It uses the Flask framework to define API endpoints. The primary endpoint will likely accept POST requests with image data, preprocess the image, extract features, use the trained model to make a prediction, and return the result (e.g., "Cataract" or "Normal" along with a probability or confidence score) in a JSON format.
  
* **`preprocessing.py`:** This module contains functions for preprocessing the input eye images. This might involve resizing, normalization, noise reduction, or other techniques to ensure consistent input for the feature extraction and prediction stages.
  
* **`train_rf.py`:** This script is responsible for training a Random Forest classification model using the data in the `dataset/` directory and the feature extraction methods defined in `feature_extraction.py`. The trained model is typically saved to a file for later use by the Flask API.
  
* **`train_svm.py`:** Similar to `train_rf.py`, this script trains a Support Vector Machine (SVM) classification model. SVM is another popular machine learning algorithm often used for image classification tasks. The trained SVM model is also saved for deployment.

## Running the Backend API

To run the Flask API and make it accessible to the Cortex Vision mobile application, follow these steps:

1.  **Ensure Dependencies are Installed:** Navigate to the repository directory in your terminal and install the required Python packages. This project likely uses libraries such as Flask, scikit-learn, OpenCV (cv2), and potentially others. You can typically install them using pip:
    ```bash
    pip install Flask scikit-learn opencv-python Pillow  # Add other necessary libraries
    ```

2.  **Run the Flask Application:** Execute the `main.py` script:
    ```bash
    python main.py
    ```
    This command will start the Flask development server. You should see output indicating the server is running on a specific address and port (e.g., `http://127.0.0.1:5000/`).

3.  **API Endpoint:** The primary endpoint for receiving image data and returning predictions will be defined in `main.py`. The mobile application will need to send POST requests to this endpoint with the image data. The specific URL and request format will be detailed in the `main.py` file.

## Communication with the Mobile App

The Cortex Vision mobile application (Flutter-based) communicates with this backend using HTTP requests to the Flask API endpoints defined in `main.py`. The typical workflow is as follows:

1.  The user uploads an eye image through the mobile app.
2.  The mobile app sends the image data (likely as a multipart form or base64 encoded string) in a POST request to the backend API endpoint.
3.  The Flask API receives the image data.
4.  The backend preprocesses the image using functions from `preprocessing.py`.
5.  Features are extracted from the preprocessed image using functions from `feature_extraction.py`.
6.  The trained machine learning model (either Random Forest or SVM, or potentially an ensemble of both) is loaded.
7.  The extracted features are fed into the loaded model to generate a prediction (e.g., "Cataract" or "Normal") and possibly a confidence score.
8.  The Flask API returns the prediction result in a structured format (e.g., JSON) back to the mobile application.
9.  The mobile application then displays the prediction to the user.

## Further Development

This repository serves as the foundation for the machine learning component of Cortex Vision. Future development could involve:

* Improving the accuracy of the machine learning models by using more data, trying different architectures, or fine-tuning hyperparameters.
* Implementing more sophisticated feature extraction techniques.
* Adding logging and monitoring to the API.
* Deploying the Flask API to a production-ready environment (e.g., using Docker and a cloud platform).
* Implementing model versioning and A/B testing.
