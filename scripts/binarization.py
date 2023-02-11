import os
import cv2
import numpy as np


def apply_commom_binarize(img: np.ndarray, threshold: int) -> np.ndarray:
    """Binarize the image using the given threshold.

    Args:
        img (numpy.ndarray): The image to be binarized.
        threshold (int): The threshold to be used.

    Returns:
        numpy.ndarray: The binarized image.
    """
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]


if __name__ == '__main__':
    image_path = os.path.join('images', 'paisagem01.jpg')
    threshold = 140

    os.makedirs('results', exist_ok=True)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(img, 'results/original.png')

    binarized = apply_commom_binarize(img, threshold)
    cv2.imwrite(binarized, 'results/binarized.png')
