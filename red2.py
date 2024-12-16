import torch
import torch.nn as nn

class SharedSubDQNN(nn.Module):
    def __init__(self, input_per_lane, num_lanes, output_dim):
        super(SharedSubDQNN, self).__init__()
        self.lane_processor = nn.Sequential(
            nn.Linear(input_per_lane, 32),
            nn.ReLU(),
            nn.Linear(32, 16)
        )
        self.fc = nn.Sequential(
            nn.Linear(num_lanes * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        batch_size, num_lanes, input_per_lane = x.shape
        processed_lanes = self.lane_processor(x.view(-1, input_per_lane))
        combined = processed_lanes.view(batch_size, -1)
        return self.fc(combined)
