from matplotlib import pyplot as plt
import os
import cv2
import logging

import numpy as np
import matplotlib
matplotlib.use('TkAgg', force=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def apply_otsu(img: np.ndarray) -> np.ndarray:
    """ Binarize the image using Otsu's method.

    Args:
        img (numpy.ndarray): The image to be binarized.

    Returns:
        numpy.ndarray: The binarized image.
    """
    if img is None:
        return None

    logger.info('Applying Otsu\'s method')
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)


if __name__ == '__main__':
    image_path = os.path.abspath(os.path.join('images', 'paisagem01.jpg'))
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
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('results/original.png', img)

    threshold, img_image = apply_otsu(img)
    logger.info(f'Threshold: {threshold}')
    cv2.imwrite('results/otsu.png', img)


    histogram, bins = np.histogram(
        gray,
        256,
        [0, 256]
    )
    plt.plot(histogram)
    plt.savefig('results/histogram.png')

    plt.hist(gray.ravel(), 256, [0, 256])
    plt.savefig('results/histogram2.png')
