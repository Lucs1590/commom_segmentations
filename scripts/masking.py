import os
import cv2
import logging

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
    image_path = os.path.abspath(os.path.join('images', 'paisagem01.jpg'))

    img = read_image(image_path, results_dir)
    logger.info('Making mask with a circle')
    height, width, _ = img.shape
    mask = np.zeros((height, width), np.uint8)
    cv2.circle(mask, (250, 200), 150, (255, 255, 255), -1)
    cv2.imwrite(os.path.join(results_dir, 'mask_circle.png'), mask)

    logger.info('Making bitwise and with the mask')
    result = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(os.path.join(results_dir, 'bitwise_and_circle.png'), result)

    logger.info('Making bitwise and with the mask and a white background')
    white_back = 255 * np.ones((height, width, 3), np.uint8)
    result[mask == 0] = white_back[mask == 0]
    cv2.imwrite(os.path.join(
        results_dir, 'bitwise_and_circle_white.png'), result)

    logger.info('Making bitwise and with the mask and a background image')
    background = cv2.imread(os.path.abspath(
        os.path.join('images', 'montanha.jpg')))
    background = cv2.resize(background, (width, height))
    result[mask == 0] = background[mask == 0]
    cv2.imwrite(os.path.join(
        results_dir, 'bitwise_and_circle_background.png'), result)

    logger.info('Isolating the object')
    image_path = os.path.abspath(os.path.join('images', 'folha_ruido.jpg'))
    img = read_image(image_path, results_dir)
    height, width = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    logger.info('Threshold: %s', thresh)

    logger.info('Saving bitwise and with the black background')
    result = cv2.bitwise_and(img, img, mask=thresh)
    cv2.imwrite(os.path.join(results_dir, 'bitwise_and_folha.png'), result)

    logger.info('Saving bitwise and with the white background')
    white_back = 255 * np.ones((height, width, 3), np.uint8)
    result[thresh == 0] = white_back[thresh == 0]

    cv2.imwrite(os.path.join(
        results_dir,
        'bitwise_and_folha_white.png'
    ), result)

    logger.info('Saving bitwise and with the background image')
    background = cv2.imread(os.path.abspath(
        os.path.join('images', 'montanha.jpg')))
    background = cv2.resize(background, (width, height))
    result[thresh == 0] = background[thresh == 0]
    cv2.imwrite(os.path.join(
        results_dir,
        'bitwise_and_folha_background.png'
    ), result)
