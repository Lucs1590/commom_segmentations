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
    logger.info('Applying gausian blur')
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    adaptive = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        3
    )
    cv2.imwrite(os.path.join(results_dir, 'adaptative_gaussian_blur.png'), adaptive)

    logger.info('Applying erosion and dilation')
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(adaptive, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)
    cv2.imwrite(os.path.join(results_dir, 'erodion.png'), erosion)
    cv2.imwrite(os.path.join(results_dir, 'dilation.png'), dilation)

    contours, hierarchy = cv2.findContours(
        dilation,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if len(contour) >= 5 and area > 1000:
            elipse = cv2.fitEllipse(contour)
            cv2.ellipse(img, elipse, (0, 255, 0), 3)

    cv2.imwrite(os.path.join(results_dir, 'contours.png'), img)
