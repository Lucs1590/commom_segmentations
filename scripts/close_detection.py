import os
import cv2
import logging

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    image_path = os.path.abspath(os.path.join('images', 'moedas02.jpg'))
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)

    with open(image_path, 'rb') as f:
        image_content = f.read()
    img = cv2.imdecode(
        np.frombuffer(image_content, np.uint8),
        cv2.IMREAD_UNCHANGED
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    canny = cv2.Canny(blur, 80, 140)

    logger.info('Applying closing')
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(canny, kernel, iterations=14)
    erosion = cv2.erode(dilated, kernel, iterations=14)

    cv2.imwrite(os.path.join(results_dir, 'canny_closing.png'), erosion)

    logger.info('Applying contour detection')
    dilated = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)
    contours, hierarchy = cv2.findContours(
        dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )

    logger.info('Filling contours')
    mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=2)
    cv2.imwrite(os.path.join(results_dir, 'filled_contours.png'), mask)
