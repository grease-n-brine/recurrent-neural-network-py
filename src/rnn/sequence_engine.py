import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import urllib.request
import os
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).parent.parent.parent

data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/airline-passengers.csv"
data_path = PROJECT_ROOT / "data/airline-passengers.csv"

if not data_path.exists():
    os.makedirs(data_path.parent, exist_ok=True)
    urllib.request.urlretrieve(data_url, data_path)

df = pd.read_csv(data_path)
raw = df[["Passengers"]].values.astype(np.float32)

passengers_raw = df['Passengers'].values.astype(np.float32)
passengers_min = passengers_raw.min()
passengers_max = passengers_raw.max()

passengers_normalized = (passengers_raw - passengers_min) / (passengers_max - passengers_min)

def create_sequences(data, seq_length):
    xs, ys = [], []

    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)
