import torch
import torch.nn as nn
from src.models.DefaultModel import DefaultBinaryModel

class CustomCNN(DefaultBinaryModel):
    def __init__(self, lr=1e-3, weight_decay=1e-4, dropout_p=0.0):
        super().__init__(lr=lr, weight_decay=weight_decay)
        self.dropout_p = dropout_p
        
        # Increased convolutional layers with more filters
        self.conv1 = nn.Conv2d(1, 64, 3, stride=1, padding=1) 
        self.conv2 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, 3, stride=1, padding=1)

        # Smaller linear layers
        self.fc1 = nn.Linear(512 * 12 * 14, 64)  # Adjusted for the output size of the last convolutional layer
        self.fc2 = nn.Linear(64, 1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        x = x[:, 0].unsqueeze(1)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.pool(nn.functional.relu(self.conv4(x)))
        x = x.flatten(start_dim=1)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
