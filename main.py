import cv2
import numpy as np
import warnings
from evaluation import evaluate_model, plot_metrics
from preprocessing import load_images_from_folder, unsharp_mask
from train_rf import train_tuned_random_forest, train_default_random_forest
from train_svm import train_svm_default, train_hyperparameter_tuned_svm

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

# Train Default SVM models
default_svm_linear, default_params_linear = train_svm_default(X_train, y_train, 'linear')
default_svm_poly, default_params_poly = train_svm_default(X_train, y_train, 'poly')
default_svm_rbf, default_params_rbf = train_svm_default(X_train, y_train, 'rbf')

# Train Tuned SVM models
tuned_svm_linear, best_params_linear = train_hyperparameter_tuned_svm(X_train, y_train, 'linear')
tuned_svm_poly, best_params_poly = train_hyperparameter_tuned_svm(X_train, y_train, 'poly')
tuned_svm_rbf, best_params_rbf = train_hyperparameter_tuned_svm(X_train, y_train, 'rbf')

# Train Default and Tuned Random Forest models
default_rf_model, default_best_params_rf = train_default_random_forest(X_train, y_train)
tuned_rf_model, tuned_best_params_rf = train_tuned_random_forest(X_train, y_train)


# Print Parameters
print("========= DEFAULT SVM PARAMETERS =========")
print("Linear SVM:", default_params_linear)
print("Polynomial SVM:", default_params_poly)
print("RBF SVM:", default_params_rbf)

print("========= TUNED SVM PARAMETERS =========")
print("Linear SVM:", best_params_linear)
print("Polynomial SVM:", best_params_poly)
print("RBF SVM:", best_params_rbf)

print("========= RANDOM FOREST PARAMETERS =========")
print("Default RF:", default_best_params_rf)
print("Tuned RF:", tuned_best_params_rf)

# Evaluate SVM & RF models on Training Data
train_default_acc_linear = default_svm_linear.score(X_train, y_train)
train_default_acc_poly = default_svm_poly.score(X_train, y_train)
train_default_acc_rbf = default_svm_rbf.score(X_train, y_train)

train_tuned_acc_linear = tuned_svm_linear.score(X_train, y_train)
train_tuned_acc_poly = tuned_svm_poly.score(X_train, y_train)
train_tuned_acc_rbf = tuned_svm_rbf.score(X_train, y_train)

# Now we can safely call .score() on the trained models
train_default_acc_rf = default_rf_model.score(X_train, y_train)
train_tuned_acc_rf = tuned_rf_model.score(X_train, y_train)


# Evaluate SVM & RF models on Test Data
default_acc_linear, default_df_linear = evaluate_model(default_svm_linear, X_test, y_test, "Linear SVM")
default_acc_poly, default_df_poly = evaluate_model(default_svm_poly, X_test, y_test, "Polynomial SVM")
default_acc_rbf, default_df_rbf = evaluate_model(default_svm_rbf, X_test, y_test, "RBF SVM")

tuned_acc_linear, tuned_df_linear = evaluate_model(tuned_svm_linear, X_test, y_test, "Linear SVM")
tuned_acc_poly, tuned_df_poly = evaluate_model(tuned_svm_poly, X_test, y_test, "Polynomial SVM")
tuned_acc_rbf, tuned_df_rbf = evaluate_model(tuned_svm_rbf, X_test, y_test, "RBF SVM")

default_acc_rf, default_df_rf = evaluate_model(default_rf_model, X_test, y_test, "Random Forest")
tuned_acc_rf, tuned_df_rf = evaluate_model(tuned_rf_model, X_test, y_test, "Random Forest")

# Print Accuracy Results
print("========= MODEL ACCURACY RESULTS =========")
print(f"Default Linear SVM - Train: {train_default_acc_linear * 100:.2f}% | Test: {default_acc_linear * 100:.2f}%")
print(f"Default Polynomial SVM - Train: {train_default_acc_poly * 100:.2f}% | Test: {default_acc_poly * 100:.2f}%")
print(f"Default RBF SVM - Train: {train_default_acc_rbf * 100:.2f}% | Test: {default_acc_rbf * 100:.2f}%")

print(f"Tuned Linear SVM - Train: {train_tuned_acc_linear * 100:.2f}% | Test: {tuned_acc_linear * 100:.2f}%")
print(f"Tuned Polynomial SVM - Train: {train_tuned_acc_poly * 100:.2f}% | Test: {tuned_acc_poly * 100:.2f}%")
print(f"Tuned RBF SVM - Train: {train_tuned_acc_rbf * 100:.2f}% | Test: {tuned_acc_rbf * 100:.2f}%")

print(f"Default Random Forest - Train: {train_default_acc_rf * 100:.2f}% | Test: {default_acc_rf * 100:.2f}%")
print(f"Tuned Random Forest - Train: {train_tuned_acc_rf * 100:.2f}% | Test: {tuned_acc_rf * 100:.2f}%")

# Visualize results
plot_metrics(default_df_linear, default_df_poly, default_df_rbf, default_df_rf)
plot_metrics(tuned_df_linear, tuned_df_poly, tuned_df_rbf, tuned_df_rf)

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
default_pred_linear = default_svm_linear.predict(user_img)
default_pred_poly = default_svm_poly.predict(user_img)
default_pred_rbf = default_svm_rbf.predict(user_img)
default_pred_rf = default_rf_model.predict(user_img)

tuned_pred_linear = tuned_svm_linear.predict(user_img)
tuned_pred_poly = tuned_svm_poly.predict(user_img)
tuned_pred_rbf = tuned_svm_rbf.predict(user_img)
tuned_pred_rf = tuned_rf_model.predict(user_img)

# Print Predictions
print("\n========= PREDICTIONS FOR USER IMAGE =========")
print("Default Models:")
print(f"  Linear SVM: {'Normal' if default_pred_linear == 0 else 'Cataract'}")
print(f"  Polynomial SVM: {'Normal' if default_pred_poly == 0 else 'Cataract'}")
print(f"  RBF SVM: {'Normal' if default_pred_rbf == 0 else 'Cataract'}")
print(f"  Random Forest: {'Normal' if default_pred_rf == 0 else 'Cataract'}")

print("\nTuned Models:")
print(f"  Tuned Linear SVM: {'Normal' if tuned_pred_linear == 0 else 'Cataract'}")
print(f"  Tuned Polynomial SVM: {'Normal' if tuned_pred_poly == 0 else 'Cataract'}")
print(f"  Tuned RBF SVM: {'Normal' if tuned_pred_rbf == 0 else 'Cataract'}")
print(f"  Tuned Random Forest: {'Normal' if tuned_pred_rf == 0 else 'Cataract'}")
