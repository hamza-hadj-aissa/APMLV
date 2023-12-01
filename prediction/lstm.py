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

LOOKBACK = 6  # look to LOOKBACK*10 minutes back
device = 'cpu'
scaler = MinMaxScaler(feature_range=(-1, 1))
oneHotEncode = LabelEncoder()


def create_sequences(dataframe, n_steps):
    for i in range(1, n_steps+1):
        dataframe[f"used_space(t-{i})"] = dataframe["used_space"].shift(i)
    dataframe.dropna(inplace=True)
    return dataframe


def data_preprocessing(df: pd.DataFrame, features):
    # set index as timestamps
    df.index = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')
    # select features
    df = df[features]

    unique_uuids = df["uuid"].unique()
    train_data = {}
    test_data = {}
    for uuid in unique_uuids:
        df_uuid = df[df['uuid'] == uuid]
        # sequence data
        df_sequence = create_sequences(df_uuid[["used_space"]], LOOKBACK)
        # scaling
        df_to_np = df_sequence.to_numpy()
        # Scaling "used_space"
        df_scaled = scaler.fit_transform(df_to_np)

        x = dc(np.flip(df_scaled[:, 1:], axis=1))
        y = df_scaled[:, 0]
        split_index = int(len(x) * 0.8)
        X_train, X_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        X_train = X_train.reshape((-1, LOOKBACK, 1))
        X_test = X_test.reshape((-1, LOOKBACK, 1))

        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))

        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()
        train_data[uuid] = (X_train, y_train)
        test_data[uuid] = (X_test, y_test)
    return train_data, test_data


class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size,
                         self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size,
                         self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0
    early_stopping_patience = 5
    early_stopping_counter = 0
    best_val_loss = float('inf')
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 'min', patience=early_stopping_patience, factor=0.5, verbose=True)

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
            # Learning rate scheduler step
            scheduler.step(avg_loss_across_batches)
            # Early stopping check
            if avg_loss_across_batches < best_val_loss:
                best_val_loss = avg_loss_across_batches
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= early_stopping_patience:
                print("Early stopping after {} epochs without improvement.".format(
                    early_stopping_patience))
                break


def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')


def eval_new_dataset(model):
    test_df = pd.read_csv(
        "/home/hamza/Desktop/studio/python/lvm_balancer/data_generation/dataset/test_logical_volume.csv")
    split_index = int(test_df.shape[0] * 0.8)
    print(test_df.iloc[:6*6*6].iloc[:split_index].iloc[4:])
    split_index = int(test_df.shape[0] * 0.8)
    train_data, test_data = data_preprocessing(
        test_df.iloc[:6*6*6], ["used_space", "uuid"])
    # train_dataset = TimeSeriesDataset(*train_data[uuid])
    test_dataset = TimeSeriesDataset(*test_data[uuid])
    test_pred = model(test_dataset.X.to(
        device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((len(test_pred), LOOKBACK+1))
    dummies[:, 0] = test_pred
    dummies = scaler.inverse_transform(dummies)
    pred = dc(dummies[:, 0])
    # actual test samples
    dummies = np.zeros((test_dataset.X.shape[0], LOOKBACK+1))
    dummies[:, 0] = test_dataset.y.flatten()
    dummies = scaler.inverse_transform(dummies)
    actual = dc(dummies[:, 0])
    return actual, pred


if __name__ == "__main__":
    df = pd.read_csv("/home/hamza/Desktop/studio/python/lvm_balancer/data_generation/dataset/logical_volume_usage_history.csv",
                     )
    train_data, test_data = data_preprocessing(
        df,
        features=["used_space", "uuid"]
    )
    for uuid in train_data.keys():
        train_dataset = TimeSeriesDataset(*train_data[uuid])
        test_dataset = TimeSeriesDataset(*test_data[uuid])

        batch_size = 16
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        for _, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            break

        model = LSTM(1, 4, 1)
        model.to(device)
        # model parameters
        learning_rate = 0.01
        num_epochs = 5
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            train_one_epoch()
            validate_one_epoch()

        model.eval()
        with torch.no_grad():
            predicted = model(train_dataset.X.to(device)).to('cpu').numpy()
        # predict test samples
        test_predictions = model(
            test_dataset.X.to(device)).detach().cpu().numpy().flatten()
        dummies = np.zeros((test_dataset.X.shape[0], LOOKBACK+1))
        dummies[:, 0] = test_predictions
        dummies = scaler.inverse_transform(dummies)
        test_predictions = dc(dummies[:, 0])

        # actual test samples
        dummies = np.zeros((test_dataset.X.shape[0], LOOKBACK+1))
        dummies[:, 0] = test_dataset.y.flatten()
        dummies = scaler.inverse_transform(dummies)
        new_y_test = dc(dummies[:, 0])
        lv_name = df[df["uuid"] == uuid]["name"].drop_duplicates().to_list()[0]
        split_index = int(len(test_dataset.X) * 0.8)

        plt.plot(
            new_y_test[:6*6], label=f'Actual usage {lv_name} (Testing)')
        plt.plot(test_predictions[:6*6],
                 label=f'Predicted usage {lv_name} (Testing)')
        plt.xlabel('Time')
        plt.ylabel('Usage')
        plt.legend()
        actual, pred = eval_new_dataset(model)
        plt.plot(actual, label="actual usage (New Dataset)")
        plt.plot(pred, label="predictions (New Dataset)")
        plt.legend()
        break
    plt.show()
