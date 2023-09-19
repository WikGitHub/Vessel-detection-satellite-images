import torch.nn as nn
import torch.nn.functional as F

from models.base_model.main import ImageClassificationBase


class BaseCnn(ImageClassificationBase):
    """
    Implementation of a simple CNN architecture with a few convolutional and connected layers. This model is going to be used as a base to compare other models to.
    """

    def __init__(self):
        """
        Initialise the model by setting up the layers.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, data):
        """
        Forward pass through the network.
        :param data: input data to the network.
        :return: output of the network.
        """
        data = F.relu(self.conv1(data))
        data = self.pool(data)
        data = data.view(-1, 16 * 40 * 40)
        data = F.relu(self.fc1(data))
        data = self.fc2(data)
        return data
