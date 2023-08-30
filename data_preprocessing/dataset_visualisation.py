import os

from matplotlib import pyplot as plt
from matplotlib.image import imread

dataset_path = "/Users/wiktoria/Desktop/Python Projects/vessel-detection-satellite-images/data/vessel_imgs"
dataset = os.listdir(dataset_path)


def visualise_dataset():
    """
    Visualise images in the provided dataset.
    :return: visualisation of the dataset.
    """

    plt.figure(figsize=(20, 7))
    plt.figtext(
        0.5, 0.95, "Visualisation of given data.", fontsize=16, ha="center", color="red"
    )
    for i in range(15):
        plt.subplot(5, 5, i + 1)
        image = imread(os.path.join(dataset_path, dataset[i]))
        plt.imshow(image)
        plt.axis("off")
        gs = plt.gca().get_gridspec()
        gs.update(top=0.85)
    plt.show()


if __name__ == "__main__":
    visualise_dataset()
