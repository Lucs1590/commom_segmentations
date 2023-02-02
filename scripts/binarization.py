import os
import cv2


def apply_commom_binarize(img, threshold=127):
    """Binarize the image using the given threshold.

    Args:
        img (numpy.ndarray): The image to be binarized.
        threshold (int): The threshold to be used.

    Returns:
        numpy.ndarray: The binarized image.
    """
    return cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]


if __name__ == '__main__':
    image_path = os.path.join('images', 'paisagem01.jpg')
    output_path = os.path.join('results', 'binarized.png')
    threshold = 127

    os.makedirs('results', exist_ok=True)

    img = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    binarized = apply_commom_binarize(img, threshold)
    cv2.imwrite(output_path, binarized)
