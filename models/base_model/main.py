import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.testing._internal.common_quantization import accuracy

from src.common._logging.main import get_logger

_logger = get_logger(__name__)


class ImageClassificationBase(nn.Module):
    """
    Basic template for implementing neural network models for image classification.
    """

    def calculate_training_loss(self, batch: torch.Tensor):
        """
        Calculate the loss for a batch of training data. The loss is calculated using cross-entropy.
        :param batch: images and labels for a batch of training data.
        :return: loss for the batch.
        """
        images, labels = batch
        predicted_scores = self(images)
        training_loss = F.cross_entropy(predicted_scores, labels)
        return training_loss

    def calculate_validation_metrics(self, batch: torch.Tensor):
        """
        Calculate the loss and accuracy for a batch of validation data.
        :param batch: images and labels for a batch of validation data.
        :return: dictionary containing the loss and accuracy for the batch.
        """
        val_images, val_labels = batch
        predicted_scores = self(val_images)
        val_loss = F.cross_entropy(predicted_scores, val_labels)
        val_accuracy = accuracy(predicted_scores, val_labels)
        return {
            "validation_loss": val_loss.detach(),
            "validation_accuracy": val_accuracy,
        }

    def calculate_average_validation_metrics(self, validation_outputs: list[dict]):
        """
        Calculate the average validation loss and accuracy for a validation epoch.
        :param validation_outputs: List of outputs from validation_step() for each batch.
        :return: A dictionary containing the average validation loss and accuracy for the epoch.
        """
        batch_losses = [x["validation_loss"] for x in validation_outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accuracies = [x["validation_accuracy"] for x in validation_outputs]
        epoch_accuracy = torch.stack(batch_accuracies).mean()
        return {
            "average_validation_loss": epoch_loss.item(),
            "average_validation_accuracy": epoch_accuracy.item(),
        }

    def log_epoch_results(self, epoch_number: int, results: dict):
        """
        Log the results of a validation epoch.
        :param epoch_number: The current epoch number.
        :param results: A dictionary containing training loss, validation loss, and validation accuracy.
        :return: None
        """
        _logger.info(
            f"Epoch {epoch_number + 1}: Training loss: {results['training_loss']}, "
            f"Validation loss: {results['average_validation_loss']}, "
            f"Validation accuracy: {results['average_validation_accuracy']}"
        )


if __name__ == "__main__":
    pass
