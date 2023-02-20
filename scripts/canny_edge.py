import os
import cv2
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    image_path = os.path.abspath(os.path.join('images', 'paisagem01.jpg'))
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    with open(image_path, 'rb') as f:
        image_content = f.read()
    img = cv2.imdecode(
        np.frombuffer(image_content, np.uint8),
        cv2.IMREAD_UNCHANGED
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    logger.info('Applying Canny')
    canny = cv2.Canny(blur, 80, 140)

    cv2.imwrite(os.path.join(results_dir, 'canny.png'), canny)

    logger.info('Applying Canny (dilated)')
    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8))
    cv2.imwrite(os.path.join(results_dir, 'canny_dilated.png'), dilated)
