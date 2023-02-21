import os
import logging

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def read_image(image_path: str, results_dir: str) -> np.ndarray:
    """Reads an image from a path and returns a numpy array.

    Args:
        image_path (str): Path to the image.
        results_dir (str): Path to the results directory.

    Returns:
        np.ndarray: Numpy array with the image.
    """
    os.makedirs(results_dir, exist_ok=True)

    with open(image_path, 'rb') as f:
        image_content = f.read()
    img = cv2.imdecode(
        np.frombuffer(image_content, np.uint8),
        cv2.IMREAD_UNCHANGED
    )

    return img


if __name__ == '__main__':
    results_dir = 'results'
    image_path = os.path.abspath(os.path.join('images', 'frutas.jpg'))
    logger.info(f'Reading image from {image_path}')
    img = read_image(image_path, results_dir)

    vectorized = img.reshape((-1, 3))
    vectorized = np.float32(vectorized)
    logger.info(f'Vectorized image shape: {vectorized.shape}')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ranges = [i for i in range(2, 11)]

    for i in ranges:
        logger.info(f'Running k-means with {i} clusters')
        ret, label, center = cv2.kmeans(
            vectorized,
            i,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))

        cv2.imwrite(os.path.join(results_dir, f'kmeans_{i}K.jpg'), res2)
