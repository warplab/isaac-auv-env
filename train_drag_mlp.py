"""
MLP to calculate drag forces and torque given CUREE velocity vector.
Input: x, y, z velocity components
Output: Fx, Fy, Fz (Force components), Mx, My, Mz (Torque Components)

Author: Steven Roche (rochesh@mit.edu)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import os


def train_mlp(data_path, input_size, output_size, batch_size, 
             epochs, learning_rate, val_size, test_size, test=False):
    """
    Train a simple MLP regressor on the provided dataset using a traditional train/validation/test split.

    Parameters:
        data_path (str): Path to the CSV data file.
        input_size (int): Number of input features.
        output_size (int): Number of output targets.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        val_size (float): Fraction of data to use for validation (from the original dataset).
        test_size (float): Fraction of data to use for test (from the remaining data after validation split).
        test (bool): If True, evaluate and print test set loss after training.

    Behavior:
        - Loads and preprocesses the data (including dropping specified columns/rows).
        - Splits the data into train, validation, and test sets.
        - Standardizes features and targets based on the training set.
        - Trains an MLP regressor, monitoring training and validation loss.
        - Plots training and validation loss curves using Plotly.
        - Optionally evaluates and prints test set loss after training.
    """
    DATA_PATH = data_path
    INPUT_SIZE = input_size 
    OUTPUT_SIZE = output_size 
    BATCH_SIZE = batch_size
    EPOCHS = epochs 
    LEARNING_RATE = learning_rate 
    VAL_SIZE = val_size
    TEST_SIZE = test_size

    # 1. Load data
    raw = pd.read_csv(DATA_PATH, header=0)
    raw.drop(columns=['v'], inplace=True)
    raw.drop([1995], axis=0, inplace=True)
    X = raw.iloc[:, :INPUT_SIZE].values.astype(np.float32)
    y = raw.iloc[:, INPUT_SIZE:INPUT_SIZE+OUTPUT_SIZE].values.astype(np.float32)

    # 2. Train/val/test split (70/15/15)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=VAL_SIZE, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=TEST_SIZE, random_state=42)

    # 3. Standardize features 
    scaler_X = StandardScaler().fit(X_train)
    scaler_y = StandardScaler().fit(y_train)
    X_train = scaler_X.transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)
    y_train = scaler_y.transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    # 4. PyTorch Dataset
    class DragDataset(Dataset):
        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_ds = DragDataset(X_train, y_train)
    val_ds = DragDataset(X_val, y_val)
    test_ds = DragDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # 5. Simple MLP Model 
    def make_mlp(input_size, output_size):
        """
        Construct a simple feedforward neural network (MLP) with one hidden layer.

        Parameters:
            input_size (int): Number of input features.
            output_size (int): Number of output targets.

        Returns:
            nn.Sequential: The constructed MLP model.
        """
        return nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    model = make_mlp(INPUT_SIZE, OUTPUT_SIZE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses = []
    val_losses = []

    # 6. Training loop with validation
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        epoch_loss = running_loss / len(train_ds)
        train_losses.append(epoch_loss)

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for xb, yb in val_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                val_loss += loss.item() * xb.size(0)
            val_loss /= len(val_ds)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    # 7. Plot training and validation loss
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=train_losses, mode='lines', name='Train Loss'))
    fig.add_trace(go.Scatter(y=val_losses, mode='lines', name='Validation Loss'))
    fig.update_layout(title='Training and Validation Loss', xaxis_title='Epoch', yaxis_title='MSE Loss')
    fig.show()

    # 8. Final test evaluation 
    if test:
        model.eval()
        with torch.no_grad():
            test_loss = 0.0
            for xb, yb in test_loader:
                preds = model(xb)
                loss = criterion(preds, yb)
                test_loss += loss.item() * xb.size(0)
            test_loss /= len(test_ds)
            print(f"Test MSE Loss: {test_loss:.4f}")

def main():
    """
    Main entry point for the script. Sets default parameters and calls the training function.
    """
    # Parameters
    DATA_PATH = 'dragData.txt'  # CSV file
    INPUT_SIZE = 3
    OUTPUT_SIZE = 6
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    VAL_SIZE = 0.15
    TEST_SIZE = .15/0.85
    test = False
    train_mlp(DATA_PATH, INPUT_SIZE, OUTPUT_SIZE, BATCH_SIZE, EPOCHS,
                LEARNING_RATE, VAL_SIZE, TEST_SIZE, test)

if __name__ == "__main__":
    main() 