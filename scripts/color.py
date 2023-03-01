import os
import logging

import cv2
import numpy as np

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, colors
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


def apply_rgb_segmentation(img, results_dir):
    logger.info('RGB segmentation')

    min_color = np.array([90, 10, 0], dtype=np.uint8)
    max_color = np.array([255, 180, 40], dtype=np.uint8)
    mask = cv2.inRange(img, min_color, max_color)
    cv2.imwrite(os.path.join(results_dir, 'mask_rgb.png'), mask)

    logger.info('Saving image with color segmentation')
    color_segment = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(os.path.join(
        results_dir, 'color_segment_rgb.png'), color_segment)

    overlay = cv2.addWeighted(img, 0.8, color_segment, 0.2, 0)
    cv2.imwrite(os.path.join(results_dir, 'overlay_rgb.png'), overlay)

    color_intervals = [
        ([90, 10, 0], [255, 180, 40]),  # red
        ([0, 80, 0], [120, 255, 120]),  # green
        ([8, 8, 160], [120, 120, 255]),  # blue
        ([0, 150, 200], [10, 255, 255]),  # yellow
        ([0, 80, 240], [80, 165, 255])  # orange
    ]

    logger.info('Segmenting image by many colors')
    for i, (min_color, max_color) in enumerate(color_intervals):
        min_color = np.array(min_color, dtype=np.uint8)
        max_color = np.array(max_color, dtype=np.uint8)
        mask = cv2.inRange(img, min_color, max_color)
        cv2.imwrite(os.path.join(results_dir, f'mask_{i}_rgb.png'), mask)

        color_segment = cv2.bitwise_and(img, img, mask=mask)
        cv2.imwrite(
            os.path.join(results_dir, f'color_segment_{i}_rbg.png'),
            color_segment
        )

        overlay = cv2.addWeighted(img, 0.8, color_segment, 0.2, 0)
        cv2.imwrite(os.path.join(results_dir, f'overlay_{i}.png'), overlay)

    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r, g, b = cv2.split(rgb)

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(
        r.flatten(),
        g.flatten(),
        b.flatten(),
        facecolors=pixel_colors,
        marker="."
    )
    axis.set_xlabel("Red")
    axis.set_ylabel("Green")
    axis.set_zlabel("Blue")
    plt.savefig(os.path.join(results_dir, 'color_histogram_rgb.png'))


def apply_hsv_segmentation(img, results_dir):
    logger.info('HSV segmentation')
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    min_color = np.array([45, 80, 40], dtype=np.uint8)
    max_color = np.array([75, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, min_color, max_color)
    cv2.imwrite(os.path.join(results_dir, 'mask_hsv.png'), mask)

    logger.info('Saving image with color segmentation')
    color_segment = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(os.path.join(
        results_dir, 'color_segment_hsv.png'),
        color_segment
    )
    pixel_colors = img.reshape((np.shape(img)[0]*np.shape(img)[1], 3))
    norm = colors.Normalize(vmin=-1., vmax=1.)
    norm.autoscale(pixel_colors)
    pixel_colors = norm(pixel_colors).tolist()

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    fig = plt.figure()
    fig.set_size_inches(18.5, 10.5)
    axis = fig.add_subplot(1, 1, 1, projection="3d")
    axis.scatter(
        h.flatten(),
        s.flatten(),
        v.flatten(),
        facecolors=pixel_colors,
        marker="."
    )
    axis.set_xlabel("Hue")
    axis.set_ylabel("Saturation")
    axis.set_zlabel("Value")
    plt.savefig(os.path.join(results_dir, 'color_histogram_hsv.png'))


if __name__ == '__main__':
    results_dir = 'results'
    image_path = os.path.abspath(os.path.join('images', 'limao_amarelo.jpg'))
    logger.info(f'Reading image from {image_path}')
    img = read_image(image_path, results_dir)

    logger.info('Segmenting image by color')
    apply_hsv_segmentation(img, results_dir)
    apply_rgb_segmentation(img, results_dir)
