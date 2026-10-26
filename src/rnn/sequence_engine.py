import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import urllib.request
import os
import pathlib

data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/refs/heads/master/airline-passengers.csv"
data_path = pathlib.Path("data/airline-passengers.csv")

if not data_path.exists():
    os.makedirs(data_path.parent, exist_ok=True)
    urllib.request.urlretrieve(data_url, data_path)
