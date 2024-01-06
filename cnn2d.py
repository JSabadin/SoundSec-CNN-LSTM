# This file is derived from work originally created by Hbbbbbby (https://github.com/Hbbbbbby/EmotionRecognition_2Dcnn-lstm).
# Original License: BSD 3-Clause License (https://github.com/Hbbbbbby/EmotionRecognition_2Dcnn-lstm/blob/main/LICENSE).
# Changes were made by converting the model from TensorFlow to PyTorch, while maintaining the same structure.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model2D(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(Model2D, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_shape[0], out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.pool2 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=128)
        self.pool3 = nn.MaxPool2d(kernel_size=4, stride=4)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(num_features=128)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)

        self._to_linear = None
        self._calculate_to_linear(input_shape)

        self.lstm = nn.LSTM(input_size=self._to_linear, hidden_size=32, batch_first=True)
        self.fc = nn.Linear(32, num_classes)

    def _calculate_to_linear(self, input_shape):
        with torch.no_grad():
            self._to_linear = np.prod(self._forward_conv(torch.zeros(1, *input_shape)).size()[1:])

    def _forward_conv(self, x):
        x = self.pool1(F.elu(self.bn1(self.conv1(x))))
        x = self.pool2(F.elu(self.bn2(self.conv2(x))))
        x = self.pool3(F.elu(self.bn3(self.conv3(x))))
        x = self.pool4(F.elu(self.bn4(self.conv4(x))))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)  # Flatten the output for LSTM
        x = x.unsqueeze(1)  # Reshape for LSTM - add sequence dimension
        x, (hn, cn) = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take the output of the last time step
        return F.log_softmax(x, dim=1)

def create_model2d(input_shape, num_classes):
    model = Model2D(input_shape, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-6)
    return model, optimizer
