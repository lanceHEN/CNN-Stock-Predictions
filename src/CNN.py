import torch
import torch.nn as nn

class CNN(nn.Module):

    def __init__(self, input_dim, seq_length, n_filters=64, kernel_size=2, pool_size=2, conv_layers=3):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=n_filters, kernel_size=kernel_size, padding=1)
        self.pool1 = nn.MaxPool1d(pool_size)

        self.convs = nn.Sequential(*[nn.Conv1d(in_channels=n_filters, out_channels=n_filters, kernel_size=kernel_size, padding=1) for _ in range(conv_layers - 1)])
        self.pools = nn.Sequential(*[nn.MaxPool1d(kernel_size=pool_size) for _ in range(conv_layers - 1)])

        final_seq_length = seq_length // (pool_size**(conv_layers))

        self.fc1 = nn.Linear(n_filters*(1+final_seq_length), 128)
        #print(n_filters*final_seq_length)
        self.fc2 = nn.Linear(128, 1) # output a single price prediction

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = x.permute(0, 2, 1)  # change shape from (batch, seq_length, features) to (batch, features, seq_length)
        #print(x.shape)
        x = self.conv1(x)
        x = self.pool1(x)
        for conv, pool in zip(self.convs, self.pools):
            x = conv(x)
            x = pool(x)

        x = self.flatten(x)
        #print(x.shape)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)