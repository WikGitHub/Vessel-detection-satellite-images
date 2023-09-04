import os

import cv2
import torch

from common._logging.main import get_logger

_logger = get_logger(__name__)
dataset_path = "/Users/wiktoria/Desktop/Python Projects/vessel-detection-satellite-images/data/vessel_imgs"
dataset = os.listdir(dataset_path)


def split_dataset(full_dataset: list[str]):
    """
    Split the dataset into training and testing sets.
    :param full_dataset: path to the full dataset.
    :return: train_dataset containing 80% of the full dataset and test_dataset containing 20% of the full dataset.
    """

    train_size = int(0.8 * len(full_dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, len(full_dataset) - train_size]
    )
    _logger.info(f"The training dataset size is {len(train_dataset)}")
    _logger.info(f"The testing dataset size is {len(test_dataset)}")

    return train_dataset, test_dataset


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
            height, width, _ = img.shape

            _logger.info(f"Image {filename} has height {height} and width {width}")


if __name__ == "__main__":
    train_dataset, test_dataset = split_dataset(dataset)
    # check_image_size(dataset_path)
