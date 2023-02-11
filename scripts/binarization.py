import os
import cv2
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_binarization(img: np.ndarray, threshold: int, threshold_type: int = cv2.THRESH_BINARY) -> np.ndarray:
    """Binarize the image using the given threshold.

    Args:
        img (numpy.ndarray): The image to be binarized.
        threshold (int): The threshold to be used.
        threshold_type (int, optional): The threshold type. Defaults to cv2.THRESH_BINARY.

    Returns:
        numpy.ndarray: The binarized image.
    """
    if img is None:
        return None

    logger.info(
        f'Applying binarization with threshold {threshold} and threshold type {threshold_type}'
    )
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(img, threshold, 255, threshold_type)[1]


if __name__ == '__main__':
    threshold_types = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV,
                       cv2.THRESH_TRUNC, cv2.THRESH_TOZERO, cv2.THRESH_TOZERO_INV]
    image_path = os.path.abspath(os.path.join('images', 'paisagem01.jpg'))
    threshold = 140

    os.makedirs('results', exist_ok=True)

    with open(image_path, 'rb') as f:
        image_content = f.read()
    img = cv2.imdecode(
        np.frombuffer(
            image_content,
            np.uint8
        ), cv2.IMREAD_UNCHANGED
    )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.imwrite('results/original.png', img)

    for threshold_type in threshold_types:
        cv2.imwrite(
            f'results/binarization_{threshold_type}.png',
            apply_binarization(
                img,
                threshold,
                threshold_type
            )
        )
