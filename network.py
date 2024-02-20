import torch
import torch.nn as nn
import torch.nn.functional as F

class CLDNN(nn.Module):
    def __init__(self, num_features, num_time_frames, num_classes=7):
        super(CLDNN, self).__init__()

        # Increased number of filters
        self.conv1 = nn.Conv1d(in_channels=num_features, out_channels=512, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(512)  # Batch Normalization
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(256)  # Batch Normalization
        # Additional convolutional layers
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=256, kernel_size=5, stride=1, padding=2)
        self.bn3 = nn.BatchNorm1d(256)  # Batch Normalization
        self.conv4 = nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(128)  # Batch Normalization

        # LSTM
        self.lstm = nn.LSTM(input_size=128, hidden_size=50, batch_first=True)
        self.dropout = nn.Dropout(0.5)  # Dropout

        # The input features to the fully connected layer is the hidden size of LSTM * number of directions * number of time steps
        self.fc1 = nn.Linear(50 *   num_time_frames, num_classes)

    def forward(self, x):
        # Assuming x is of shape [batch_size, num_features, num_time_frames]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Prepare the output of the conv layers for the LSTM
        x = x.permute(0, 2, 1)  # Permute to [batch_size, num_time_frames, 128]
        x, _ = self.lstm(x)

        # Apply dropout
        x = self.dropout(x)

        # Flatten the output for the dense layers
        x = x.contiguous().view(x.size(0), -1)

        x = self.fc1(x)

        return F.log_softmax(x, dim=1)

def create_model(num_features=39, num_time_frames=251, num_classes=7):
    model = CLDNN(num_features, num_time_frames, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=1e-6)
    return model, optimizer