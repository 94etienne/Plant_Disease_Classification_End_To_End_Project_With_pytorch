# model.py
import torch
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, n_classes):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),  # 3 filters
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(3, 6, kernel_size=3, padding=1),  # 6 filters
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(6, 12, kernel_size=3, padding=1),  # 12 filters
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        # Calculate the flattened size after the convolutional layers
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 256, 256)
            dummy_output = self.features(dummy_input)
            self.flattened_size = dummy_output.view(-1).size(0)

        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 16),
            nn.ReLU(),
            nn.Linear(16, n_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
