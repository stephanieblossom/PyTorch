import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from alpha_vantage.timeseries import TimeSeries

# Replace with your Alpha Vantage API key
API_KEY = 'EUE8QLP56U9M5PK4'

# Download stock data from Alpha Vantage
symbol = 'MSFT'
ts = TimeSeries(key=API_KEY, output_format='pandas')
data, _ = ts.get_daily_adjusted(symbol=symbol, outputsize='full')
data = data['5. adjusted close'].values[::-1]

# Preprocess and create dataset
def create_dataset(data, window_size=20):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

window_size = 20
X, y = create_dataset(data, window_size=window_size)
X_train, X_test = X[:-window_size], X[-window_size:]
y_train, y_test = y[:-window_size], y[-window_size:]

# Normalize data
train_mean, train_std = X_train.mean(), X_train.std()
X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create a custom Dataset and DataLoader
class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = StockDataset(X_train, y_train)
test_dataset = StockDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# Define the feed-forward neural network
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Create the model, loss function, and optimizer
input_size = window_size
hidden_size = 64
output_size = 1
model = StockPredictor(input_size, hidden_size, output_size)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
num_epochs = 100
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluate the model and make predictions
model.eval()
with torch.no_grad():
    test_predictions = model(X_test).squeeze()

# Convert predictions back to the original scale
test_predictions = test_predictions * train_std + train_mean
y_test_original = y_test * train_std + train_mean

# Print the predicted and actual prices for comparison
for pred, actual in zip(test_predictions, y_test_original):
    print(f"Predicted: {pred.item():.2f}, Actual: {actual.item():.2f}")

# Get the most recent `window_size` days of data
latest_data = data[-window_size:]

# Normalize the input data
latest_data_normalized = (latest_data - train_mean) / train_std

# Convert the input data to a PyTorch tensor
latest_data_tensor = torch.tensor(latest_data_normalized, dtype=torch.float32).unsqueeze(0)

# Make a prediction for the next day
model.eval()
with torch.no_grad():
    next_day_prediction = model(latest_data_tensor).squeeze()

# Convert the prediction back to the original scale
next_day_prediction_original = next_day_prediction * train_std + train_mean

print(f"Next day predicted price: {next_day_prediction_original.item():.2f}")
