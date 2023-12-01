from lstm import data_preprocessing, TimeSeriesDataset, scaler, LOOKBACK, LSTM
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


learning_rate = 0.01
model = LSTM(1, 4, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


loaded_state = torch.load(
    "/home/hamza/Desktop/studio/python/lvm_balancer/prediction/models/lstm_checkpoint_4fa5dc13-ab08-44d4-a44f-5876e96ab242.pt")
model.load_state_dict(loaded_state["model_state"])
optimizer.load_state_dict(loaded_state["optimizer"])
model.eval()
df = pd.read_csv(
    "/home/hamza/Desktop/studio/python/lvm_balancer/prediction/dataset/test_logical_volume.csv")
train_data, test_data = data_preprocessing(
    df.iloc[180:360],
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


dummies = np.zeros((test_dataset.X.shape[0], LOOKBACK+1))
dummies[:, 0] = test_dataset.y.flatten()
dummies = scaler.inverse_transform(dummies)
actual = dc(dummies[:, 0])
plt.plot(test_predictions, label="Predictions")
plt.plot(actual, label="Actual")
plt.legend()
plt.show()
