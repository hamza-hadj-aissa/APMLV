import datetime
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import mean_squared_error
from root_directory import root_directory
from logs.Logger import Logger

DEVICE = 'cpu'


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
                         self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_stacked_layers, batch_size,
                         self.hidden_size).to(DEVICE)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LogicalVolumeModel():
    def __init__(self, df: pd.DataFrame, input_size: int, hidden_size: int, num_hidden_layers: int, lookback: int):
        self.df: pd.DataFrame = df
        self.lv_uuid = df["uuid"].drop_duplicates().to_list()[
            0]
        self.lv_name = df["name"].drop_duplicates().to_list()[
            0]
        self.model: LSTM = LSTM(
            input_size=input_size, hidden_size=hidden_size, num_stacked_layers=num_hidden_layers)
        self.scaler: MinMaxScaler = MinMaxScaler(feature_range=(-1, 1))
        self.lookback: int = lookback
        self.logger = Logger(
            name=f"{self.lv_name}:Model", path=f"{root_directory}/logs/model_{self.lv_uuid}.log")

    def train_model(self, Optimizer, n_epochs: int, lr: float, batch_size: int, Loss_function):
        train_data, test_data = self.__data_preprocessing(
            features=["used_space"]
        )

        train_dataset = TimeSeriesDataset(*train_data)
        test_dataset = TimeSeriesDataset(*test_data)

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False)

        self.model.to(DEVICE)
        # model parameters
        loss_function = Loss_function()
        optimizer = Optimizer(
            self.model.parameters(), lr=lr)

        for _ in range(n_epochs):
            self.__train_one_epoch(_,
                                   train_loader=train_loader, optimizer=optimizer, loss_function=loss_function, early_stopping_patience=10)
            self.__validate_one_epoch(
                test_loader=test_loader, loss_function=loss_function)
        self.__plot_results(train_dataset=train_dataset,
                            test_dataset=test_dataset, offset=36)
        model_checkpoint = {
            "epoch": n_epochs,
            "model_state": self.model.state_dict(),
            "optimizer": optimizer.state_dict()
        }
        # save model
        torch.save(model_checkpoint,
                   f"{root_directory}/prediction/models/lstm_checkpoint_{self.lv_uuid}.pt")

    def __create_sequences(self, dataframe, n_steps):
        for i in range(1, n_steps+1):
            dataframe[f"used_space(t-{i})"] = dataframe["used_space"].shift(i)
        dataframe.dropna(inplace=True)
        return dataframe

    def __data_preprocessing(self, features):
        # set index as timestamps
        self.df.index = pd.to_datetime(
            self.df['date'], format='%Y-%m-%d %H:%M:%S')
        # select features
        df = self.df[features].copy()

        train_data = {}
        test_data = {}
        # sequence data
        df_sequence = self.__create_sequences(df, self.lookback)
        # scaling
        df_to_np = df_sequence.to_numpy()
        # Scaling "used_space"
        df_scaled = self.scaler.fit_transform(df_to_np)
        x = dc(np.flip(df_scaled[:, 1:], axis=1))
        y = df_scaled[:, 0]
        split_index = int(len(x) * 0.8)
        X_train, X_test = x[:split_index], x[split_index:]
        y_train, y_test = y[:split_index], y[split_index:]
        X_train = X_train.reshape((-1, self.lookback, 1))
        X_test = X_test.reshape((-1, self.lookback, 1))

        y_train = y_train.reshape((-1, 1))
        y_test = y_test.reshape((-1, 1))

        X_train = torch.tensor(X_train).float()
        y_train = torch.tensor(y_train).float()
        X_test = torch.tensor(X_test).float()
        y_test = torch.tensor(y_test).float()
        train_data = (X_train, y_train)
        test_data = (X_test, y_test)
        return train_data, test_data

    def __train_one_epoch(self, _, train_loader, optimizer, loss_function, early_stopping_patience: int):
        self.model.train(True)
        self.logger.get_logger().info(f'Epoch: {_ + 1}')
        running_loss = 0.0
        early_stopping_counter = 0
        best_val_loss = float('inf')
        # Learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, 'min', patience=early_stopping_patience, factor=0.5, verbose=True)
        for batch_index, batch in enumerate(train_loader):
            x_batch, y_batch = batch[0].to(DEVICE), batch[1].to(DEVICE)

            output = self.model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # display model performance every every 100 sample
            if batch_index % 100 == 99:
                avg_loss_across_batches = running_loss / 100
                self.logger.get_logger().info('Batch {0} | Loss: {1:.5f} '.format(batch_index+1,
                                                                                  avg_loss_across_batches,
                                                                                  ))
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
                    self.logger.get_logger().info("Early stopping after {} epochs without improvement.".format(
                        early_stopping_patience))
                    break

    def __validate_one_epoch(self, test_loader, loss_function):
        self.model.train(False)
        running_loss = 0.0

        for _, batch in enumerate(test_loader):
            x_batch, y_batch = batch[0].to(DEVICE), batch[1].to(DEVICE)
            self.model.eval()
            with torch.no_grad():
                output = self.model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

        avg_loss_across_batches = running_loss / len(test_loader)

        self.logger.get_logger().info(
            'Val Loss: {0:.5f}'.format(avg_loss_across_batches))
        self.logger.get_logger().info('***************************************************')

    def __inverse_scaling(self, matrix, train_dataset: TimeSeriesDataset, test_dataset: TimeSeriesDataset, split_index):
        dummies = np.zeros(
            (train_dataset.y.shape[0] + test_dataset.y.shape[0], self.lookback+1))
        dummies[:split_index, 0] = matrix
        dummies = self.scaler.inverse_transform(dummies)
        return dc(dummies[:, 0])

    def __inverse_scaling_test(self, matrix, train_dataset: TimeSeriesDataset, test_dataset: TimeSeriesDataset, split_index):
        dummies = np.zeros(
            (train_dataset.y.shape[0] + test_dataset.y.shape[0], self.lookback+1))
        dummies[split_index:, 0] = matrix
        dummies = self.scaler.inverse_transform(dummies)
        return dc(dummies[:, 0])

    def __plot_results(self, train_dataset: TimeSeriesDataset, test_dataset: TimeSeriesDataset, offset: int = 36):
        self.model.eval()
        with torch.no_grad():
            # training predictions
            predicted_train = self.model(
                train_dataset.X.to(DEVICE)).detach().cpu().numpy().flatten()
            # testing predictions
            test_predictions = self.model(
                test_dataset.X.to(DEVICE)).detach().cpu().numpy().flatten()
        # logical volume dataset split index
        split_index = train_dataset.y.shape[0]
        # inverse train actual data scaling
        train_actual = self.__inverse_scaling(
            train_dataset.y.flatten(), train_dataset, test_dataset, split_index)
        # inverse train prediction data scaling
        train_predictions = self.__inverse_scaling(
            predicted_train, train_dataset, test_dataset, split_index)
        # inverse test actual data scaling
        test_actual = self.__inverse_scaling_test(
            test_dataset.y.flatten(), train_dataset, test_dataset, split_index)
        # inverse test prediction data scaling
        test_predictions = self.__inverse_scaling_test(
            test_predictions, train_dataset, test_dataset, split_index)

        # array of NaN, size of offset
        zerros_array = np.ones(offset) * np.nan

        # extract datetime for x axis
        x_axis = self.df[["date"]].iloc[self.lookback:].iloc[split_index -
                                                             offset:split_index+offset].to_numpy()
        x_axis = np.flip(x_axis.flatten())
        datetime_values = [datetime.datetime.strptime(
            dt_str, '%Y-%m-%d %H:%M:%S') for dt_str in x_axis]
        x_axis = [dt.strftime('%m/%d %H:%M') for dt in datetime_values]

        plt.figure(figsize=(16, 8))
        plt.plot(x_axis[:36*2], np.concatenate([train_actual[split_index-offset:split_index], zerros_array]),
                 label=f"Actual usage {self.lv_name} (Training)")
        plt.plot(x_axis[:36*2], np.concatenate([train_predictions[split_index-offset:split_index], zerros_array]),
                 label=f"Predicted usage {self.lv_name} (Training)")
        plt.plot(
            x_axis, np.concatenate([zerros_array, test_actual[split_index:split_index+offset]]), label=f'Actual usage {self.lv_name} (Testing)')
        plt.plot(x_axis, np.concatenate([zerros_array, test_predictions[split_index:split_index+offset]]),
                 label=f'Predicted usage {self.lv_name} (Testing)')
        plt.xlabel('Time')
        plt.xticks(rotation=90, fontsize=8)
        plt.yticks(minor=True)
        plt.ylabel('Usage (MiB)')
        plt.legend()
        plt.grid(True, which='both')
        plt.savefig(
            f"{root_directory}/prediction/figures/model_{self.lv_uuid}.png", dpi=300, bbox_inches="tight")
        test_MAE = mean_squared_error(
            test_actual[split_index:], test_predictions[split_index:], squared=False)
        train_MAE = mean_squared_error(
            train_actual[:split_index], train_predictions[:split_index], squared=False)
        test_MSE = mean_squared_error(
            test_actual[split_index:], test_predictions[split_index:], squared=True)
        train_MSE = mean_squared_error(
            train_actual[:split_index], train_predictions[:split_index], squared=True)
        self.logger.get_logger().info(
            f"Train MAE: {train_MAE} | Test MAE: {test_MAE}\nTrain MSE: {train_MSE} | Test MSE: {test_MSE}")


if __name__ == "__main__":
    df = pd.read_csv(f"{root_directory}/prediction/dataset/logical_volume_usage_history.csv",
                     )
    for uuid in df["uuid"].drop_duplicates():
        logicalVolumeModel = LogicalVolumeModel(
            df[df["uuid"] == uuid], input_size=1, hidden_size=32, num_hidden_layers=3, lookback=6)
        logicalVolumeModel.train_model(Optimizer=torch.optim.Adam,
                                       n_epochs=100, lr=0.00001, batch_size=16, Loss_function=nn.SmoothL1Loss)
    plt.show()
