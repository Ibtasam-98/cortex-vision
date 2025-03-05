
import cv2
import numpy as np
import os


def unsharp_mask(image, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
    return sharpened


def load_images_from_folder(folder, img_size=(100, 100)):
    """Loads images from a folder, resizes them, applies sharpening, and normalizes."""
    images = []
    labels = []
    for label in os.listdir(folder):
        class_path = os.path.join(folder, label)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                image = cv2.imread(img_path)
                if image is None:
                    continue
                image = cv2.resize(image, img_size)
                image = image / 255.0  # Normalize pixel values
                image = unsharp_mask(image)
                images.append(image)
                labels.append(0 if label.lower() == 'normal' else 1)

    return np.array(images), np.array(labels)
