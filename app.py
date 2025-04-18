from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the trained models (replace with your actual model paths)
try:
    default_svm_linear = joblib.load('model/svm/default_svm_linear.joblib')
    default_svm_poly = joblib.load('model/svm/default_svm_poly.joblib')
    default_svm_rbf = joblib.load('model/svm/default_svm_rbf.joblib')
    tuned_svm_linear = joblib.load('model/svm/tuned_svm_linear.joblib')
    tuned_svm_poly = joblib.load('model/svm/tuned_svm_poly.joblib')
    tuned_svm_rbf = joblib.load('model/svm/tuned_svm_rbf.joblib')
    default_rf_model = joblib.load('model/rf/default_rf_model.joblib')
    tuned_rf_model = joblib.load('model/rf/tuned_rf_model.joblib')
    print(f"[{datetime.now()}] Models loaded successfully.")
except FileNotFoundError as e:
    error_message = f"[{datetime.now()}] Error: One or more model files not found: {e}. Please ensure the model files are saved correctly in the same directory as app.py or provide the correct paths."
    print(error_message)
    models_loaded = False
else:
    models_loaded = True

IMG_SIZE = (100, 100)

def preprocess_image(image_path, img_size=(100, 100)):
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"[{datetime.now()}] Error: Could not load image from path: {image_path}")
            return None
        image = cv2.resize(image, img_size)
        image = image / 255.0
        image = unsharp_mask(image)
        return image.reshape(1, -1)
    except Exception as e:
        print(f"[{datetime.now()}] Error preprocessing image: {e}")
        return None

def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened

@app.route('/predict', methods=['POST'])
def predict():
    if not models_loaded:
        return jsonify({'error': 'Models not loaded. Prediction unavailable.'}), 503

    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    temp_path = 'temp_image.jpg'
    try:
        # Save the uploaded image temporarily
        image_file.save(temp_path)
        print(f"[{datetime.now()}] Received and saved image: {image_file.filename} to {temp_path}")

        processed_image = preprocess_image(temp_path, IMG_SIZE)
        os.remove(temp_path)  # Clean up temporary file
        print(f"[{datetime.now()}] Preprocessed image and removed temporary file.")

        if processed_image is None:
            return jsonify({'error': 'Failed to preprocess image'}), 500

        predictions = {}

        # Make predictions with all models
        try:
            pred_linear_default = default_svm_linear.predict(processed_image)[0]
            prob_linear_default = np.max(default_svm_linear.predict_proba(processed_image)) * 100
            predictions['Default Linear SVM'] = {
                'detection': 'Cataract' if pred_linear_default == 1 else 'Normal',
                'accuracy': f'{prob_linear_default:.2f}%'
            }
        except Exception as e:
            predictions['Default Linear SVM'] = {'error': f'Prediction failed: {e}'}

        try:
            pred_poly_default = default_svm_poly.predict(processed_image)[0]
            prob_poly_default = np.max(default_svm_poly.predict_proba(processed_image)) * 100
            predictions['Default Polynomial SVM'] = {
                'detection': 'Cataract' if pred_poly_default == 1 else 'Normal',
                'accuracy': f'{prob_poly_default:.2f}%'
            }
        except Exception as e:
            predictions['Default Polynomial SVM'] = {'error': f'Prediction failed: {e}'}

        try:
            pred_rbf_default = default_svm_rbf.predict(processed_image)[0]
            prob_rbf_default = np.max(default_svm_rbf.predict_proba(processed_image)) * 100
            predictions['Default RBF SVM'] = {
                'detection': 'Cataract' if pred_rbf_default == 1 else 'Normal',
                'accuracy': f'{prob_rbf_default:.2f}%'
            }
        except Exception as e:
            predictions['Default RBF SVM'] = {'error': f'Prediction failed: {e}'}

        try:
            pred_rf_default = default_rf_model.predict(processed_image)[0]
            prob_rf_default = default_rf_model.predict_proba(processed_image)[0][1] * 100 # Probability of class 1
            predictions['Default Random Forest'] = {
                'detection': 'Cataract' if pred_rf_default == 1 else 'Normal',
                'accuracy': f'{prob_rf_default:.2f}%'
            }
        except Exception as e:
            predictions['Default Random Forest'] = {'error': f'Prediction failed: {e}'}

        try:
            pred_linear_tuned = tuned_svm_linear.predict(processed_image)[0]
            prob_linear_tuned = np.max(tuned_svm_linear.predict_proba(processed_image)) * 100
            predictions['Tuned Linear SVM'] = {
                'detection': 'Cataract' if pred_linear_tuned == 1 else 'Normal',
                'accuracy': f'{prob_linear_tuned:.2f}%'
            }
        except Exception as e:
            predictions['Tuned Linear SVM'] = {'error': f'Prediction failed: {e}'}

        try:
            pred_poly_tuned = tuned_svm_poly.predict(processed_image)[0]
            prob_poly_tuned = np.max(tuned_svm_poly.predict_proba(processed_image)) * 100
            predictions['Tuned Polynomial SVM'] = {
                'detection': 'Cataract' if pred_poly_tuned == 1 else 'Normal',
                'accuracy': f'{prob_poly_tuned:.2f}%'
            }
        except Exception as e:
            predictions['Tuned Polynomial SVM'] = {'error': f'Prediction failed: {e}'}

        try:
            pred_rbf_tuned = tuned_svm_rbf.predict(processed_image)[0]
            prob_rbf_tuned = np.max(tuned_svm_rbf.predict_proba(processed_image)) * 100
            predictions['Tuned RBF SVM'] = {
                'detection': 'Cataract' if pred_rbf_tuned == 1 else 'Normal',
                'accuracy': f'{prob_rbf_tuned:.2f}%'
            }
        except Exception as e:
            predictions['Tuned RBF SVM'] = {'error': f'Prediction failed: {e}'}

        try:
            pred_rf_tuned = tuned_rf_model.predict(processed_image)[0]
            prob_rf_tuned = tuned_rf_model.predict_proba(processed_image)[0][1] * 100 # Probability of class 1
            predictions['Tuned Random Forest'] = {
                'detection': 'Cataract' if pred_rf_tuned == 1 else 'Normal',
                'accuracy': f'{prob_rf_tuned:.2f}%'
            }
        except Exception as e:
            predictions['Tuned Random Forest'] = {'error': f'Prediction failed: {e}'}

        print(f"[{datetime.now()}] Prediction results: {predictions}")
        return jsonify(predictions)

    except Exception as e:
        error_message = f"[{datetime.now()}] An error occurred during prediction: {e}"
        print(error_message)
        return jsonify({'error': error_message}), 500
    finally:
        # Ensure the temporary file is removed even if an error occurs
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    print(f"[{datetime.now()}] Starting Flask API...")
    app.run(debug=True, host='0.0.0.0', port=5000)