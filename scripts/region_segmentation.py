import os
import logging

import cv2
from skimage.color import rgb2gray
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


def flatten_image(img: np.ndarray) -> np.ndarray:
    """Flattens a 3D image to a 1D array.

    Args:
        img (np.ndarray): Image to be flattened.

    Returns:
        np.ndarray: Flattened image.
    """
    height, width = img.shape[:2]
    img_flat = img.reshape(height * width)
    return img_flat


def segment_back_foward_ground(img: np.ndarray) -> np.ndarray:
    """Segments an image in background, foreground and ground.

    Args:
        img (np.ndarray): Image to be segmented.

    Returns:
        np.ndarray: Segmented image.
    """
    img_flat = flatten_image(img.copy())
    mean = img_flat.mean()

    num_pixels = len(img_flat)

    logger.info('Making segmentation based on mean threshold')
    for i in range(num_pixels):
        if img_flat[i] < mean:
            img_flat[i] = 0
        else:
            img_flat[i] = 255

    img_segmented = img_flat.reshape(img.shape[0], img.shape[1])
    return img_segmented


def segment_image_regions(input_image: np.ndarray, num_regions: int = 3) -> np.ndarray:
    """Segments an image in regions.

    Args:
        img (np.ndarray): Image to be segmented.
        num_regions (int): Number of regions to segment the image.

    Returns:
        np.ndarray: Segmented image.
    """
    flat_img = flatten_image(input_image.copy())

    ranges = np.linspace(0, 1, num_regions + 1)
    ranges = ranges[:-1]
    values = list(map(int, np.linspace(0, 255, num_regions)))

    num_pixels = len(flat_img)

    logger.info('Making segmentation based on mean threshold')
    for i in range(num_pixels):
        for j in range(num_regions):
            if j == num_regions - 1:
                flat_img[i] = values[j]
                break
            if flat_img[i] >= ranges[j] and flat_img[i] < ranges[j+1]:
                flat_img[i] = values[j]
                break

    img_segmented = flat_img.reshape(img.shape[0], img.shape[1])
    return img_segmented


if __name__ == '__main__':
    results_dir = 'results'
    image_path = os.path.abspath(os.path.join('images', 'paisagem01.jpg'))

    img = read_image(image_path, results_dir)

    height, width = img.shape[:2]
    gray = rgb2gray(img)

    segmented_img = segment_back_foward_ground(gray)
    cv2.imwrite(
        os.path.join(results_dir, 'segmented.png'),
        segmented_img
    )

    logger.info('Saving segmented image with regions')
    segmented_img_regions = segment_image_regions(gray, 3)
    cv2.imwrite(
        os.path.join(results_dir, 'segmented_regions.png'),
        segmented_img_regions
    )
