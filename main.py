import cv2
import numpy as np
import warnings
from evaluation import evaluate_model, plot_metrics
from preprocessing import load_images_from_folder, unsharp_mask
from sklearn.metrics import classification_report

from train_rf import train_random_forest
from train_svm import train_svm

warnings.filterwarnings("ignore")


TRAIN_FOLDER = 'dataset/train'
TEST_FOLDER = 'dataset/test'
IMG_SIZE = (100, 100)

# Load datasets
X_train, y_train = load_images_from_folder(TRAIN_FOLDER, IMG_SIZE)
X_test, y_test = load_images_from_folder(TEST_FOLDER, IMG_SIZE)

# Reshape data for SVM (Flatten the images)
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)

# Train SVM models
svm_linear, best_params_linear = train_svm(X_train, y_train, 'linear')
svm_poly, best_params_poly = train_svm(X_train, y_train, 'poly')
svm_rbf, best_params_rbf = train_svm(X_train, y_train, 'rbf')

# Train Random Forest model
rf_model, best_params_rf = train_random_forest(X_train, y_train)

print("===================================")
print("Best Linear SVM Params:", best_params_linear)
print("Best Polynomial SVM Params:", best_params_poly)
print("Best RBF SVM Params:", best_params_rbf)
print("Best Random Forest Params:", best_params_rf)
print("===================================")

# Evaluate models on training data
train_acc_linear = svm_linear.score(X_train, y_train)
train_acc_poly = svm_poly.score(X_train, y_train)
train_acc_rbf = svm_rbf.score(X_train, y_train)
train_acc_rf = rf_model.score(X_train, y_train)

# Evaluate models on testing data
acc_linear, df_linear = evaluate_model(svm_linear, X_test, y_test, "Linear SVM")
acc_poly, df_poly = evaluate_model(svm_poly, X_test, y_test, "Polynomial SVM")
acc_rbf, df_rbf = evaluate_model(svm_rbf, X_test, y_test, "RBF SVM")
acc_rf, df_rf = evaluate_model(rf_model, X_test, y_test, "Random Forest")


# Print training and testing accuracy
print("===================================")
print(f"Linear SVM Training Accuracy: {train_acc_linear * 100:.2f}%")
print(f"Linear SVM Testing Accuracy: {acc_linear * 100:.2f}%\n")

print(f"Polynomial SVM Training Accuracy: {train_acc_poly * 100:.2f}%")
print(f"Polynomial SVM Testing Accuracy: {acc_poly * 100:.2f}%\n")

print(f"RBF SVM Training Accuracy: {train_acc_rbf * 100:.2f}%")
print(f"RBF SVM Testing Accuracy: {acc_rbf * 100:.2f}%\n")

print(f"Random Forest Training Accuracy: {train_acc_rf * 100:.2f}%")
print(f"Random Forest Testing Accuracy: {acc_rf * 100:.2f}%")

print("===================================")

# Visualize results
plot_metrics(df_linear, df_poly, df_rbf, df_rf)

# Load user image for prediction
user_input_path = input("Enter image path: ")
user_img = cv2.imread(user_input_path)

if user_img is None:
    print("Error loading image. Check file path.")
    exit()

user_img = cv2.resize(user_img, IMG_SIZE)
user_img = user_img / 255.0
user_img = unsharp_mask(user_img)
user_img = user_img.reshape(1, -1)

# Predictions
pred_linear = svm_linear.predict(user_img)
pred_poly = svm_poly.predict(user_img)
pred_rbf = svm_rbf.predict(user_img)
pred_rf = rf_model.predict(user_img)

print("===================================")
print("\nPredictions for user image:")
print("Linear SVM:", "Normal" if pred_linear == 0 else "Cataract")
print("Polynomial SVM:", "Normal" if pred_poly == 0 else "Cataract")
print("RBF SVM:", "Normal" if pred_rbf == 0 else "Cataract")
print("Random Forest:", "Normal" if pred_rf == 0 else "Cataract")

print("===================================")

print(f"Linear SVM Accuracy: {acc_linear * 100:.2f}%")
print(f"Polynomial SVM Accuracy: {acc_poly * 100:.2f}%")
print(f"RBF SVM Accuracy: {acc_rbf * 100:.2f}%")
print(f"Random Forest Accuracy: {acc_rf * 100:.2f}%")
print("===================================")
