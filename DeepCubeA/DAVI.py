import gzip
import pickle

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloader(batch_size, dataset_type='train'):
    class SokobanDataset(Dataset):
        def __init__(self, room_structures, states, distances):
            self.data = []
            self.target = distances

            for state, room_structure in zip(states, room_structures):
                wall_map = torch.from_numpy((state == 0).astype(int)).flatten()
                target_map = torch.from_numpy((room_structure == 2).astype(int)).flatten()
                boxes_map = torch.from_numpy((state == 4).astype(int)).flatten()
                agent_map = torch.from_numpy((state == 5).astype(int)).flatten()

                self.data.append(torch.cat((wall_map, target_map, boxes_map, agent_map), 0))

        def __len__(self):
            return len(self.target)

        def __getitem__(self, idx):
            return self.data[idx], self.target[idx]

    with gzip.open(f'../data/{dataset_type}/room_structures_{dataset_type}.pkl.gz', 'rb') as f:
        room_structures = pickle.load(f)

    with gzip.open(f'../data/{dataset_type}/states_{dataset_type}.pkl.gz', 'rb') as f:
        states = pickle.load(f)

    with gzip.open(f'../data/{dataset_type}/distances_{dataset_type}.pkl.gz', 'rb') as f:
        distances = pickle.load(f)

    sokoban_dataset = SokobanDataset(room_structures, states, distances)
    print('sokoban_dataset =', len(sokoban_dataset))

    return DataLoader(sokoban_dataset, batch_size=batch_size)


def plot_distribution():
    with gzip.open('../data/train/distances_train.pkl.gz', 'rb') as f:
        distances = pickle.load(f)

    stats = {}
    for distance in distances:
        if distance in stats:
            stats[distance] += 1
        else:
            stats[distance] = 1
    print('train', stats)
    plt.bar(range(len(stats)), list(stats.values()), align='center', label='train data')
    plt.xticks(range(len(stats)), list(stats.keys()))

    with gzip.open('../data/test/distances_test.pkl.gz', 'rb') as f:
        distances = pickle.load(f)

    stats = {}
    for distance in distances:
        if distance in stats:
            stats[distance] += 1
        else:
            stats[distance] = 1
    print('test', stats)
    plt.bar(range(len(stats)), list(stats.values()), align='center', label='test data')

    plt.xlabel('steps to solution')
    plt.ylabel('# examples')
    plt.legend()
    plt.show()


def create_model():
    class ResidualBlock(nn.Module):
        expansion = 1

        def __init__(self, num_features):
            super(ResidualBlock, self).__init__()

            self.fc1 = nn.Linear(num_features, num_features)
            self.bn1 = nn.BatchNorm1d(num_features)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Linear(num_features, num_features)
            self.bn2 = nn.BatchNorm1d(num_features)

        def forward(self, x):
            identity = x

            out = self.fc1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.fc2(out)
            out = self.bn2(out)

            out += identity

            return self.relu(out)

    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()

            self.model = nn.Sequential(
                nn.Linear(400, 5000),
                nn.BatchNorm1d(5000),
                nn.ReLU(),

                nn.Linear(5000, 1000),
                nn.BatchNorm1d(1000),
                nn.ReLU(),

                ResidualBlock(1000),
                ResidualBlock(1000),
                ResidualBlock(1000),
                ResidualBlock(1000),

                nn.Linear(1000, 1)
            )

        def forward(self, x):
            return self.model(x)

    return Net()


plot_distribution()

batch_size = 128
data_loader_train = create_dataloader(batch_size, 'train')
data_loader_test = create_dataloader(batch_size, 'test')
model = create_model().to(device)
print(summary(model, (400,)))

optimizer = optim.Adam(model.parameters())
loss = nn.MSELoss()

epochs = 25
hold_train_loss = []
hold_test_loss = []

fx, tr_y = None, None
for epoch in tqdm(range(epochs), desc='Epoch'):
    train_loss = []
    test_loss = []
    for batch_idx, (data, target) in enumerate(data_loader_train):
        tr_x, tr_y = data.float().to(device), target.float().to(device)

        # Reset gradient
        optimizer.zero_grad()

        # Forward pass
        fx = model(tr_x)
        output = loss(fx, tr_y.view(-1, 1))  # loss for this batch
        train_loss.append(output.detach().cpu().numpy())

        # Backward
        output.backward()

        # Update parameters based on backprop
        optimizer.step()

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader_test):
            tr_x, tr_y = data.float().to(device), target.float().to(device)
            fx = model(tr_x)
            test_loss.append(loss(fx, tr_y.view(-1, 1)).detach().cpu().numpy())  # loss for this batch

    hold_train_loss.append(np.mean(train_loss))
    hold_test_loss.append(np.mean(test_loss))

print(fx)
print(tr_y)

plt.plot(np.array(hold_train_loss), label='train')
plt.plot(np.array(hold_test_loss), label='train')
plt.show()

print(hold_train_loss)
