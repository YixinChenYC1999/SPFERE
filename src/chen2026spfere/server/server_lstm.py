import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from importlib import resources

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(LSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size1, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size1, hidden_size2, batch_first=True)
        self.fc1 = nn.Linear(hidden_size2, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, output_size)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(device_type):
    model = LSTMModel(input_size=2, hidden_size1=100, hidden_size2=50, output_size=1)
    with resources.path("chen2026spfere.lstm", f"lstm_{device_type}.pt") as p:
        model.load_state_dict(torch.load(p))
    model.eval()
    return model

def predict_voltage(device_type, rough_window_data, global_min=2.6, global_max=4.2):
    
    model = load_model(device_type)
    model.eval()
    
    data_series = pd.Series(rough_window_data)
    
    Normalized_Voltage = (data_series - global_min) / (global_max - global_min)
    Voltage_Change_Rate = data_series.diff().fillna(0) / (global_max - global_min) * 10
    
    features = np.stack((Normalized_Voltage, Voltage_Change_Rate), axis=1)
    X_input = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        pred = model(X_input).item()

    predicted_voltage = pred * (global_max - global_min) + global_min
    return predicted_voltage
