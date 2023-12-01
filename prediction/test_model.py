from lstm import data_preprocessing, TimeSeriesDataset, scaler, LOOKBACK
import pandas as pd
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau


model = torch.load(
    "/home/hamza/Desktop/studio/python/lvm_balancer/data_generation/models/model_4fa5dc13-ab08-44d4-a44f-5876e96ab242.pth")

df = pd.read_csv(
    "/home/hamza/Desktop/studio/python/lvm_balancer/data_generation/dataset/test_logical_volume.csv")
train_data, test_data = data_preprocessing(
    df,
    features=["used_space", "uuid"]
)
batch_size = 16
train_dataset = TimeSeriesDataset(
    *train_data["4fa5dc13-ab08-44d4-a44f-5876e96ab242"])
test_dataset = TimeSeriesDataset(
    *test_data["4fa5dc13-ab08-44d4-a44f-5876e96ab242"])


test_predictions = model(
    test_dataset.X.to("cpu")).detach().cpu().numpy().flatten()
dummies = np.zeros((test_dataset.X.shape[0], LOOKBACK+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)
test_predictions = dc(dummies[:, 0])
