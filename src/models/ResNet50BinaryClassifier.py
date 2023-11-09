import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models

from torchmetrics.classification import BinaryPrecision, BinaryRecall


class ResNet50BinaryClassifier(pl.LightningModule):
    """
    Binary classifier based on ResNet50 architecture.

    This module uses a pre-trained ResNet50 model to extract features from input images,
    and adds a binary classification head on top of it. The classifier is trained to
    distinguish between two classes: "burst" and "no_burst".

    Args:
        None

    Attributes:
        precision (BinaryPrecision): Binary precision metric.
        recall (BinaryRecall): Binary recall metric.
        test_labels (List[Tensor]): List of binary labels for test set.
        test_preds (List[Tensor]): List of binary predictions for test set.
        val_labels (List[Tensor]): List of binary labels for validation set.
        val_preds (List[Tensor]): List of binary predictions for validation set.

    Methods:
        forward: Computes forward pass of the model.
        training_step: Computes loss and logs metrics during training.
        test_step: Computes loss and logs metrics during testing.
        on_test_epoch_end: Computes and logs precision and recall after testing.
        validation_step: Computes predictions and logs metrics during validation.
        on_validation_epoch_end: Computes and logs precision and recall after validation.
        configure_optimizers: Configures the optimizer used during training.
    """


class ResNet50BinaryClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Sequential(nn.Linear(num_features, 1), nn.Sigmoid())

        for param in self.resnet50.parameters():
            param.requires_grad = False

        layers_to_train = ["layer3", "layer4", "avgpool", "fc"]
        for name, child in self.resnet50.named_children():
            if name in layers_to_train:
                for param in child.parameters():
                    param.requires_grad = True

        self.precision = BinaryPrecision(threshold=0.5)
        self.recall = BinaryRecall(threshold=0.5)

        self.train_labels = []
        self.train_preds = []
        self.val_labels = []
        self.val_preds = []
        self.test_labels = []
        self.test_preds = []

    def forward(self, x):
        return self.resnet50(x)

    def __step(self, batch):
        images, info = batch

        binary_labels = [0 if label == "no_burst" else 1 for label in info["label"]]
        binary_labels = torch.tensor(binary_labels).float().view(-1, 1)
        binary_labels = binary_labels.to(images.device)

        images = images.expand(-1, 3, -1, -1)
        outputs = self(images)
        return outputs, binary_labels

    def training_step(self, batch, batch_idx):
        outputs, binary_labels = self.__step(batch)
        loss = nn.BCELoss()(outputs, binary_labels)

        self.train_labels.append(binary_labels)
        self.train_preds.append(outputs)

        self.log("train_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss
    
    def on_train_epoch_end(self):
        train_labels = torch.cat(self.train_labels, dim=0)
        train_preds = torch.cat(self.train_preds, dim=0)

        train_precision = self.precision(train_preds, train_labels)
        train_recall = self.recall(train_preds, train_labels)

        self.log("train_precision", train_precision, prog_bar=True, logger=True)
        self.log("train_recall", train_recall, prog_bar=True, logger=True)

        self.train_labels = []
        self.train_preds = [] 
    
    def validation_step(self, batch, batch_idx):
        outputs, binary_labels = self.__step(batch)
        loss = nn.BCELoss()(outputs, binary_labels)

        predictions = (outputs >= 0.5).int()
        self.val_labels.append(binary_labels.int())
        self.val_preds.append(predictions)

        self.log("val_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss
    
    def on_validation_epoch_end(self):
        val_labels = torch.cat(self.val_labels, dim=0)
        val_preds = torch.cat(self.val_preds, dim=0)

        val_precision = self.precision(val_preds, val_labels)
        val_recall = self.recall(val_preds, val_labels)

        self.log("val_precision", val_precision, prog_bar=True, logger=True)
        self.log("val_recall", val_recall, prog_bar=True, logger=True)

        self.val_labels = []
        self.val_preds = []

    def test_step(self, batch, batch_idx):
        outputs, binary_labels = self.__step(batch)
        loss = nn.BCELoss()(outputs, binary_labels)

        self.test_labels.append(binary_labels)
        self.test_preds.append(outputs)

        self.log("test_loss", loss, prog_bar=True, logger=True, batch_size=len(batch))
        return loss

    def on_test_epoch_end(self):
        test_labels = torch.cat(self.test_labels, dim=0)
        test_preds = torch.cat(self.test_preds, dim=0)

        test_precision = self.precision(test_preds, test_labels)
        test_recall = self.recall(test_preds, test_labels)

        self.log("test_precision", test_precision, prog_bar=True, logger=True)
        self.log("test_recall", test_recall, prog_bar=True, logger=True)

        self.test_labels = []
        self.test_preds = []

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
