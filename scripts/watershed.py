import os
import logging

import cv2
import numpy as np

from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from matplotlib import pyplot as plt

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


def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Calculates the euclidean distance between two points.

    Args:
        p1 (tuple): First point.
        p2 (tuple): Second point.

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


if __name__ == '__main__':
    results_dir = 'results'
    image_path = os.path.abspath(os.path.join('images', 'moedas03.jpg'))
    logger.info(f'Reading image from {image_path}')
    img = read_image(image_path, results_dir)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    logger.info('Applying threshold')
    thresh_val, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )
    logger.info('Threshold value: %s', str(thresh_val))

    dilated = cv2.dilate(thresh, np.ones((3, 3), np.uint8), iterations=2)
    segmented = cv2.erode(dilated, np.ones((3, 3), np.uint8), iterations=2)
    cv2.imwrite(os.path.join(results_dir, 'thresh.png'), segmented)

    logger.info('Applying watershed')
    distance_map = ndi.distance_transform_edt(segmented)
    plt.imshow(distance_map)
    plt.savefig(os.path.join(results_dir, 'distance_map.png'))

    local_max = peak_local_max(
        distance_map,
        indices=False,
        min_distance=20,
        labels=segmented
    )
    markers = ndi.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-distance_map, markers, mask=segmented)
    logger.info('Number of coins: %s', str(len(np.unique(labels)) - 1))

    logger.info('Saving results')
    plt.clf()
    plt.imshow(labels)
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plot = plt.imshow(labels)
    plt.colorbar(plot)
    plt.savefig(os.path.join(results_dir, 'labels_watershed.png'))
