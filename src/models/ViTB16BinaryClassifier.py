import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import torchvision.models as models
from torchvision.models import vit_b_16

from torchmetrics.classification import BinaryPrecision, BinaryRecall

class ViTB16BinaryClassifier(pl.LightningModule):
    def __init__(self, lr=1e-4, weight_decay=1e-4):
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        
        # ViT B-16 Modell laden
        self.vit_b_16 = vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

        # Parameter freezen
        for param in self.vit_b_16.parameters():
            param.requires_grad = False

        # Die letzten Encoder-Layer unfreezen
        for layer in [self.vit_b_16.encoder.layers.encoder_layer_10, 
                      self.vit_b_16.encoder.layers.encoder_layer_11]:
            for param in layer.parameters():
                param.requires_grad = True

        # Output zu binär umwandeln
        self.vit_b_16.heads = nn.Sequential(nn.Linear(768, 1), nn.Sigmoid())

        # Metriken
        self.precision = BinaryPrecision(threshold=0.5)
        self.recall = BinaryRecall(threshold=0.5)

        # Label- und Vorhersagelisten
        self.train_loss = []
        self.train_labels = []
        self.train_preds = []
        self.val_loss = []
        self.val_labels = []
        self.val_preds = []
        self.test_loss = []
        self.test_labels = []
        self.test_preds = []

    def forward(self, x):
        return self.vit_b_16(x)
    
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
        self.train_loss.append(loss)
        return loss
    
    def on_train_epoch_end(self):
        train_labels = torch.cat(self.train_labels, dim=0)
        train_preds = torch.cat(self.train_preds, dim=0)

        train_precision = self.precision(train_preds, train_labels)
        train_recall = self.recall(train_preds, train_labels)

        self.log("train_loss", torch.stack(self.train_loss).sum(), prog_bar=True, logger=True)
        self.log("train_precision", train_precision, prog_bar=True, logger=True)
        self.log("train_recall", train_recall, prog_bar=True, logger=True)

        self.train_labels = []
        self.train_preds = [] 
        self.train_loss = []
    
    def validation_step(self, batch, batch_idx):
        outputs, binary_labels = self.__step(batch)
        loss = nn.BCELoss()(outputs, binary_labels)

        predictions = (outputs >= 0.5).int()
        self.val_labels.append(binary_labels.int())
        self.val_preds.append(predictions)
        self.val_loss.append(loss)
        return loss
    
    def on_validation_epoch_end(self):
        val_labels = torch.cat(self.val_labels, dim=0)
        val_preds = torch.cat(self.val_preds, dim=0)

        val_precision = self.precision(val_preds, val_labels)
        val_recall = self.recall(val_preds, val_labels)

        self.log("val_loss", torch.stack(self.val_loss).sum(), prog_bar=True, logger=True)
        self.log("val_precision", val_precision, prog_bar=True, logger=True)
        self.log("val_recall", val_recall, prog_bar=True, logger=True)

        self.val_labels = []
        self.val_preds = []
        self.val_loss = []

    def test_step(self, batch, batch_idx):
        outputs, binary_labels = self.__step(batch)
        loss = nn.BCELoss()(outputs, binary_labels)

        self.test_labels.append(binary_labels)
        self.test_preds.append(outputs)
        self.test_loss.append(loss)
        return loss

    def on_test_epoch_end(self):
        test_labels = torch.cat(self.test_labels, dim=0)
        test_preds = torch.cat(self.test_preds, dim=0)

        test_precision = self.precision(test_preds, test_labels)
        test_recall = self.recall(test_preds, test_labels)

        self.log("test_loss", torch.stack(self.test_loss).sum(), prog_bar=True, logger=True)
        self.log("test_precision", test_precision, prog_bar=True, logger=True)
        self.log("test_recall", test_recall, prog_bar=True, logger=True)

        self.test_labels = []
        self.test_preds = []
        self.test_loss = []

    def configure_optimizers(self):
        return optim.SGD(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)