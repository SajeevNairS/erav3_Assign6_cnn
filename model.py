import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(16)  # Stabilize learning
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=10, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(10)
        
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # First spatial reduction

        self.conv4 = nn.Conv2d(in_channels=10, out_channels=16, kernel_size=3, padding=0)
        self.bn4 = nn.BatchNorm2d(16) 
        self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0)
        self.bn5 = nn.BatchNorm2d(16)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=0)
        self.bn6 = nn.BatchNorm2d(16)
        self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(16)
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # Changed to AdaptiveAvgPool2d for better GAP implementation
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=10, kernel_size=1, padding=0)

        self.dropout = nn.Dropout(0.05)  # Light regularization
        self.dropout1 = nn.Dropout(0.06)
        self.dropout2 = nn.Dropout(0.06)
        self.dropout3 = nn.Dropout(0.01)
        self.dropout4 = nn.Dropout(0.05)
        self.dropout5 = nn.Dropout(0.05)
        self.dropout6 = nn.Dropout(0.05)
        self.dropout7 = nn.Dropout(0.01)
        self.dropout8 = nn.Dropout(0.005)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout2(F.relu(self.bn2(self.conv2(x))))
        x = self.conv3(x)
        x = self.pool1(x)
        
        x = self.dropout4(F.relu(self.bn4(self.conv4(x))))
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))
        x = self.dropout6(F.relu(self.bn6(self.conv6(x))))
        x = self.dropout7(F.relu(self.bn7(self.conv7(x))))
        
        x = self.gap(x)
        x = self.conv8(x)
        
        x = x.view(x.size(0), -1)
        
        return F.log_softmax(x, dim=-1)

def create_model():
    """Create and return an instance of the model"""
    return Net() 