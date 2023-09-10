import os

import cv2

from src.common._logging.main import get_logger

_logger = get_logger(__name__)
dataset_path = "/Users/wiktoria/Desktop/Python Projects/vessel-detection-satellite-images/data/raw/vessel_imgs"
dataset = os.listdir(dataset_path)


def check_image_size(dataset_path: str):
    """
    Check the size of the images in the dataset.
    :param dataset_path: path to the dataset.
    :return: log the size of the images in the dataset.
    """
    for filename in os.listdir(dataset_path):
        if filename.endswith(".png"):
            image_path = os.path.join(dataset_path, filename)
            img = cv2.imread(image_path)
            height, width, channels = img.shape

            _logger.info(
                f"Image {filename} has height {height}, a width of {width} and {channels} channels."
            )


if __name__ == "__main__":
    check_image_size(dataset_path)
