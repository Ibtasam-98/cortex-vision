
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates the model on the test set and returns performance metrics."""
    y_pred = model.predict(X_test.reshape(len(X_test), -1))
    accuracy = accuracy_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    df_report = pd.DataFrame(report).transpose()

    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred))

    return accuracy, df_report


def plot_metrics(df_report_linear, df_report_poly, df_report_rbf, df_report_rf):
    """Plots heatmaps of precision, recall, and F1-score for different models."""
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    sns.heatmap(df_report_linear[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu")
    plt.title('Linear SVM - Metrics')

    plt.subplot(1, 4, 2)
    sns.heatmap(df_report_poly[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu")
    plt.title('Polynomial SVM - Metrics')

    plt.subplot(1, 4, 3)
    sns.heatmap(df_report_rbf[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu")
    plt.title('RBF SVM - Metrics')

    plt.subplot(1, 4, 4)
    sns.heatmap(df_report_rf[['precision', 'recall', 'f1-score']], annot=True, cmap="YlGnBu")
    plt.title('Random Forest - Metrics')

    plt.tight_layout()
    plt.show()

def plot_accuracy(train_acc, test_acc, model_names):
    n_models = len(model_names)
    x = np.arange(n_models)
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, train_acc, width, label='Training Accuracy')
    rects2 = ax.bar(x + width/2, test_acc, width, label='Testing Accuracy')

    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Testing Accuracy by Model')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=45, ha="right")
    ax.legend()

    ax.bar_label(rects1, padding=3, fmt='%.2f')
    ax.bar_label(rects2, padding=3, fmt='%.2f')

    fig.tight_layout()
    plt.show()