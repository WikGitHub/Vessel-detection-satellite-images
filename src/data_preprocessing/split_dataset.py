import os

import torch

from src.common._logging.main import get_logger

_logger = get_logger(__name__)
dataset_path = "/Users/wiktoria/Desktop/Python Projects/vessel-detection-satellite-images/data/raw/vessel_imgs"
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


if __name__ == "__main__":
    train_dataset, test_dataset = split_dataset(dataset)
