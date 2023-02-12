import os
import cv2
import logging
from typing import Union

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_otsu(img: np.ndarray, convert_to_gray: bool = True) -> Union[np.ndarray, int]:
    """ Binarize the image using Otsu's method.

    Args:
        img (numpy.ndarray): The image to be binarized.

    Returns:
        numpy.ndarray: The binarized image.
    """
    if img is None:
        return None

    logger.info('Applying Otsu\'s method')
    if convert_to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

if __name__ == '__main__':
    image_path = os.path.abspath(os.path.join('images', 'livro-texto.jpg'))
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    with open(image_path, 'rb') as f:
        image_content = f.read()
    img = cv2.imdecode(
        np.frombuffer(image_content, np.uint8),
        cv2.IMREAD_UNCHANGED
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    otsu = apply_otsu(gray, False)
    cv2.imwrite(os.path.join(results_dir, 'otsu_adaptative.png'), otsu)

    logger.info('Applying adaptative threshold with mean')
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        11,
        9
    )
    cv2.imwrite(os.path.join(results_dir, 'adaptative_mean.png'), adaptive)

    logger.info('Applying adaptative threshold with gaussian')
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        9
    )
    cv2.imwrite(os.path.join(results_dir, 'adaptative_gaussian.png'), adaptive)

    