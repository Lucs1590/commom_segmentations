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

    logger.info('Applying Sobel (handcrafted)')
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    k_sobel_x = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    k_sobel_y = np.array(
        [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    sobel_x = cv2.filter2D(blur, -1, k_sobel_x)
    sobel_y = cv2.filter2D(blur, -1, k_sobel_y)
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    cv2.imwrite(os.path.join(results_dir, 'sobel_x.png'), sobel_x)
    cv2.imwrite(os.path.join(results_dir, 'sobel_y.png'), sobel_y)
    cv2.imwrite(os.path.join(results_dir, 'sobel.png'), sobel)

    # simple sobel
    logger.info('Applying Sobel (cv2)')
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    cv2.imwrite(os.path.join(results_dir, 'sobel_x_cv2.png'), sobel_x)
    cv2.imwrite(os.path.join(results_dir, 'sobel_y_cv2.png'), sobel_y)
    cv2.imwrite(os.path.join(results_dir, 'sobel_cv2.png'), sobel)
