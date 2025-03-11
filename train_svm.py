from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from feature_extraction import extract_and_visualize_features


def train_hyperparameter_tuned_svm(X_train, y_train, kernel_type):
    param_grid = {
        'linear': {'C': [0.01, 0.1, 1, 10, 100]},
        'poly': {'C': [0.01, 0.1, 1, 10, 100], 'degree': [2, 3, 4], 'gamma': ['scale', 'auto']},
        'rbf': {'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['scale', 'auto']}
    }
    model = SVC(kernel=kernel_type, probability=True)
    grid_search = GridSearchCV(model, param_grid[kernel_type], cv=5, scoring='accuracy', n_jobs=-1)

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Extract and visualize features
    extract_and_visualize_features(X_train, y_train, f"Hyperparameter-Tuned SVM ({kernel_type})")

    return best_model, grid_search.best_params_


def train_svm_default(X_train, y_train, kernel_type):
    param_dict = {
        'linear': {'C': 1.0},
        'poly': {'C': 1.0, 'degree': 3, 'gamma': 'scale'},
        'rbf': {'C': 1.0, 'gamma': 'scale'}
    }

    params = param_dict[kernel_type]
    model = SVC(kernel=kernel_type, probability=True, **params)
    model.fit(X_train, y_train)

    # Extract and visualize features
    extract_and_visualize_features(X_train, y_train, f"Default SVM ({kernel_type})")

    return model, params