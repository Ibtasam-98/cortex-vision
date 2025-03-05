import numpy as np
import cv2
import matplotlib.pyplot as plt


def extract_and_visualize_features(images, y_train, model_name):

    color_features = []
    edge_features = []
    shape_features = []

    for img in images:
        img = img.reshape(100, 100, 3)
        img_uint8 = (img * 255).astype(np.uint8)

        # **1. Color Features (HSV mean)**
        hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
        h_mean, s_mean, v_mean = np.mean(hsv, axis=(0, 1))
        color_features.append([h_mean, s_mean, v_mean])

        # **2. Edge Features (Canny Edge Detection)**
        edges = cv2.Canny(img_uint8, 100, 200)
        edge_count = np.sum(edges > 0)
        edge_features.append(edge_count)

        # **3. Shape Features (Hu Moments)**
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        moments = cv2.moments(gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        shape_features.append(hu_moments)

    color_features = np.array(color_features)
    edge_features = np.array(edge_features)
    shape_features = np.array(shape_features)

    plot_features(color_features, edge_features, shape_features, y_train, model_name)


def plot_features(color_features, edge_features, shape_features, y_train, model_name):

    plt.figure(figsize=(12, 4))
    plt.suptitle(f'Color Feature Visualization - {model_name}', fontsize=14)
    titles = ["Hue Mean", "Saturation Mean", "Value Mean"]
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(range(len(color_features)), color_features[:, i], c=y_train, cmap='coolwarm', alpha=0.6)
        plt.title(titles[i])
    plt.show()

    # **Plot Edge Features**
    plt.figure(figsize=(6, 4))
    plt.scatter(range(len(edge_features)), edge_features, c=y_train, cmap='coolwarm', alpha=0.6)
    plt.title(f"Edge Pixel Count - {model_name}")
    plt.show()

    # **Plot Shape Features (Hu Moments)**
    plt.figure(figsize=(12, 6))
    plt.suptitle(f'Shape Feature Visualization - {model_name}', fontsize=14)
    for i in range(7):
        plt.subplot(2, 4, i + 1)
        plt.scatter(range(len(shape_features)), np.array(shape_features)[:, i], c=y_train, cmap='coolwarm', alpha=0.6)
        plt.title(f"Hu Moment {i + 1}")
    plt.tight_layout()
    plt.show()
