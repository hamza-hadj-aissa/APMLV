import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from prediction.lstm import LSTM
from root_directory import root_directory


def load_model(lv_uuid: str):
    model = LSTM(1, 64, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    loaded_state = torch.load(
        f"{root_directory}/prediction/models/lstm_checkpoint_{lv_uuid}.pt")
    model.load_state_dict(loaded_state["model_state"])
    optimizer.load_state_dict(loaded_state["optimizer"])
    model.eval()
    return model


def load_scaler(lv_uuid: str) -> MinMaxScaler:
    # loaded_state = torch.load(
    #     f"{root_directory}/prediction/models/lstm_checkpoint_{lv_uuid}.pt")
    # scaler = MinMaxScaler(feature_range=(-1, 1))
    # scaler.data_min_ = loaded_state["scaler"]['data_min_']
    # scaler.data_max_ = loaded_state["scaler"]['data_max_']
    # scaler.feature_range = loaded_state["scaler"]['feature_range']
    return joblib.load(f"{root_directory}/prediction/scalers/scaler_{lv_uuid}.joblib")


# load model
model = load_model(lv_uuid="7fedf800-2af3-40bd-b836-6393fe5e1241")
# load scaler
scaler = load_scaler(lv_uuid="7fedf800-2af3-40bd-b836-6393fe5e1241")
# to reshaped numpy array
lv_fs_usage_np = np.array([173, 176, 194, 241, 239, 218, 265]).reshape((1, 7))
# transform (scale) lv fs usage
lv_fs_usage_scaled = scaler.transform(lv_fs_usage_np)
print(lv_fs_usage_scaled)
# lv_fs_usage_scaled = lv_fs_usage_scaled
# to pyTorch tensor
lv_fs_usage_tensor = torch.tensor(
    lv_fs_usage_scaled[:, :-1].reshape(-1, 6, 1)).float()
print(lv_fs_usage_tensor)
# make prediction
prediction = model(lv_fs_usage_tensor).to(
    "cpu").detach().numpy().flatten()

dummies = np.zeros((len(prediction), 6+1))
dummies[:, 0] = prediction
dummies = scaler.inverse_transform(dummies)
pred = dummies[:, 0].copy()[0]
print(int(np.floor(pred)))
