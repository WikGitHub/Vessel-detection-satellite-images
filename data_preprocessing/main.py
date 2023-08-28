import torch

dataset_path = 'data/vessel_imgs'


def split_dataset(full_dataset: str):
    """
    Split the dataset into training and testing sets.
    :param full_dataset: path to the full dataset.
    :return: train_dataset containing 80% of the full dataset and test_dataset containing 20% of the full dataset.
    """

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    return train_dataset, test_dataset


if __name__ == "__main__":
    split_dataset(dataset_path)
