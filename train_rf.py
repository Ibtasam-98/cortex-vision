from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from feature_extraction import extract_and_visualize_features


def train_tuned_random_forest(X_train, y_train):
    """Train a tuned Random Forest model using GridSearchCV."""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 5],
        'max_features': ['sqrt', 'log2'],  # Limit features for each tree (helps generalization)
        'bootstrap': [True]  # Ensure bootstrapping is enabled
    }

    model = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    extract_and_visualize_features(X_train, y_train, "Tuned Random Forest")

    return best_model, grid_search.best_params_


def train_default_random_forest(X_train, y_train):
    """Train a default Random Forest model with standard hyperparameters."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )

    model.fit(X_train, y_train)
    extract_and_visualize_features(X_train, y_train, "Default Random Forest")

    return model, model.get_params()
