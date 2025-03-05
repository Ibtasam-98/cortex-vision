from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from feature_extraction import extract_and_visualize_features


def train_svm(X_train, y_train, kernel_type):
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
    extract_and_visualize_features(X_train, y_train, kernel_type)
    return best_model, grid_search.best_params_

