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
    image_path = os.path.abspath(os.path.join('images', 'cubo_magico.jpg'))
    logger.info(f'Reading image from {image_path}')
    img = read_image(image_path, results_dir)

    logger.info('Segmenting image by color')
    min_color = np.array([90, 10, 0], dtype=np.uint8)
    max_color = np.array([255, 180, 40], dtype=np.uint8)
    mask = cv2.inRange(img, min_color, max_color)
    cv2.imwrite(os.path.join(results_dir, 'mask.png'), mask)

    logger.info('Saving image with color segmentation')
    color_segment = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(os.path.join(results_dir, 'color_segment.png'), color_segment)

    overlay = cv2.addWeighted(img, 0.8, color_segment, 0.2, 0)
    cv2.imwrite(os.path.join(results_dir, 'overlay.png'), overlay)

    color_intervals = [
        ([90, 10, 0], [255, 180, 40]),
        ([0, 80, 0], [120, 255, 120]),
        ([8, 8, 160], [120, 120, 255]),
        ([0, 150, 200], [10, 255, 255]),
        ([0, 80, 240], [80, 165, 255])
    ]

    for i, (min_color, max_color) in enumerate(color_intervals):
        min_color = np.array(min_color, dtype=np.uint8)
        max_color = np.array(max_color, dtype=np.uint8)
        mask = cv2.inRange(img, min_color, max_color)
        cv2.imwrite(os.path.join(results_dir, f'mask_{i}.png'), mask)

        color_segment = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(os.path.join(results_dir, f'color_segment_{i}.png'), color_segment)

        overlay = cv2.addWeighted(img, 0.8, color_segment, 0.2, 0)
        cv2.imwrite(os.path.join(results_dir, f'overlay_{i}.png'), overlay)