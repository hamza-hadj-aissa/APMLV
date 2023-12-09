import joblib
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from prediction.lstm import LOOKBACK, LSTM
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
    return joblib.load(f"{root_directory}/prediction/scalers/scaler_{lv_uuid}.joblib")


def reverse_transform(scaler: MinMaxScaler, scaled_prediction):
    dummies = np.zeros((len(scaled_prediction), LOOKBACK+1), dtype=float)
    dummies[:, 0] = scaled_prediction
    dummies = scaler.inverse_transform(dummies)
    pred = dummies[:, 0].copy()
    return pred
