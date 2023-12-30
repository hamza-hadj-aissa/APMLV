# APMLV: Automated Prediction and Management of Logical Volumes

`APMLV` is a project that leverages deep learning and automation to optimize the management of logical volumes resources. It consists of the following components:

-   A data extraction module that collects logical volume usage data from various sources and stores them in a SQL database
-   A data analysis and preprocessing module that cleans and transforms the data using Python
-   A machine learning module that trains an LSTM model for time series forecasting and makes predictions on future usage of logical volumes
-   An automation module that uses Ansible to adjust the logical volume configuration based on the predictions and improve the performance and efficiency of the system

## Table of Contents

-   [Project Prerequisites](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#prerequisites)
-   [Installation](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#installation)
-   [Configuration](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#configuration)
    -   [Ansible configuration](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#ansible-configuration)
-   [Project overview](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#project-overview)
    -   [Automated Prediction and Management of Logical Volumes using LSTM Recurrent Networks](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#automated-prediction-and-management-of-logical-volume-using-lstm-recurrent-networks)
    -   [Model usage](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#model-usage)
    -   [Training the model](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#training-the-model)
    -   [Database schema](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#database-schema)
    -   [Supported filesystems](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#supported-filesystems)
-   [Workflow](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#workflow)
-   [Logging](https://github.com/hamza-hadj-aissa/lvm_balancer/tree/main?tab=readme-ov-file#logging)

## Prerequisites

Before running the project, make sure the following software and dependencies are installed on your machine:

-   LVM (Logical Volume Manager)
-   Python 3.10.x
-   Pip
-   PostgreSQL
-   Ansible

## Installation

**1** - Clone this repository to your machine by running

```bash
git clone https://github.com/hamza-hadj-aissa/lvm_balancer.git
```

**2** - Set Up a Virtual Environment

```
python -m venv env
```

**3** - Install required libraries

```bash
pip install -r requirements.txt
```

## Configuration

**1** - Make a copy of .env.sample and name it .env

```bash
cp .env.sample .env
```

**2** - Open .env and fill in the values for the following variables based on your environment

```text
HOST_NAME = "localhost"
HOST_IP = "127.0.0.1"

DB_NAME = "lvm_balancer"
DB_PORT = 5432
DB_USER = "your_database_user"
DB_PASSWORD = "your_database_password"
```

-   HOST_NAME and HOST_IP: The hostname and IP address where your database is hosted.
-   DB_NAME: The name of your database.
-   DB_PORT: The port on which your database server is running.
-   DB_USER and DB_PASSWORD: Your database username and password.

### Ansible configuration

-   Create an Ansible inventory.ini file, typically named inventory, to define the hosts (virtual machines in this case) that Ansible will manage:

```bash
touch ansible/inventory.ini
```

-   Edit the inventory file and add the IP addresses or hostnames of your virtual machines:

```ini
[vm_group]
192.168.1.101
192.168.1.102
```

-   Ensure that you can SSH into your virtual machines from the machine where Ansible is installed.
    You can always test your ansible connection:

```bash
ansible -i ansible/inventory.ini vm_group -m ping
```

## Project overview

### Automated Prediction and Management of Logical Volumes using LSTM Recurrent Networks

#### Overview

In the context of lvm_balancer, the prediction of logical volumes (LV) usage is crucial for proactive management and optimization of storage resources. Traditional methods might fall short in capturing the dynamic and non-linear patterns of storage usage. To address this, we employ LSTM (Long Short-Term Memory) Recurrent Networks, a type of deep learning architecture known for its capability to model sequences and time-series data.

#### How LSTM Recurrent Networks Work

LSTM networks are a special kind of Recurrent Neural Network (RNN) that can learn and remember patterns over long sequences, making them particularly suitable for time-series data like storage usage metrics. Here's a simplified breakdown:

-   **Long-term Memory Cells:** LSTMs have memory cells that can maintain information over long sequences. This ability helps capture trends and patterns in storage usage data that traditional methods might overlook.

-   **Gates:** LSTMs have mechanisms called gates (input, forget, and output gates) that regulate the flow of information into and out of the memory cells. This gating mechanism allows LSTMs to decide what information to keep or discard, making them adept at handling sequences with long-range dependencies.

-   **Training:** The LSTM network is trained using historical storage usage data. During training, the network learns the underlying patterns and correlations in the data, enabling it to make accurate predictions.

#### Benefits

-   **Accurate Predictions:** LSTMs can capture intricate patterns and non-linear relationships in storage usage data, leading to more accurate predictions compared to traditional methods.

-   **Proactive Management:** By forecasting future storage usage, administrators can proactively allocate resources, prevent potential bottlenecks, and optimize the overall storage infrastructure.

-   **Scalability:** The LSTM model can be scaled to handle larger datasets and adapt to evolving storage usage patterns. (_It's essential to approach scaling with caution to ensure proper handling of the model_ ). Related article: [Incremental Ensemble LSTM Model towards Time Series Data](https://www.sciencedirect.com/science/article/abs/pii/S0045790621001592)

### Model usage

The LSTM-based prediction model in lvm_balancer leverages historical LV usage metrics to forecast future usage trends. By analyzing past data points, the model can provide insights into potential storage demands, enabling more informed decision-making and efficient resources allocation.

#### Decision-making Process

The allocation of logical volumes is determined through a multi-step process:

-   **Predicting Future Usage:** Utilizing LSTM Recurrent Networks, the system forecasts the future usage of each logical volume. This prediction serves as a baseline for understanding the expected growth or reduction in storage requirements.

-   **Historical Proportion Analysis:** To maintain balance and fairness within the volume group, the historical proportion of each logical volume's usage relative to its neighbors is calculated. This analysis ensures that no single logical volume disproportionately consumes resources, leading to potential bottlenecks or inefficiencies.

-   **Priority Factor Integration:** The priority factor is incorporated into the allocation factor calculation, assigning weights or rankings to logical volumes based on their priority levels. Logical volumes with higher priority factors receive preferential treatment in the allocation process.
-   **Demand-to-Space Ratio:** The calculation of the allocation factor is designed to ensure that the cumulative demands of all logical volumes are appropriately scaled relative to the available free space in the volume group. Thereby facilitating a balanced and efficient allocation strategy.

### Training the model

Each logical volume is associated with its own trained LSTM model, enabling tailored predictions and proactive storage management. This individualized approach allows lvm_balancer to account for unique usage patterns, trends, and requirements specific to each logical volume

#### Data Preparation:

-   **Dataset Location:** Ensure that the historical LV usage metrics dataset is accessible at the specified location:

```bash
prediction/dataset/logical_volume_usage_history.csv
```

#### Execute Training Script:

-   **Command Execution:** Run the provided script (lstm.py) to initiate the data preprocessing, scaling, and LSTM model training process:

```bash
python prediction/lstm.py
```

-   **Monitor Progress:** Monitor the script execution for any logs, messages, or outputs indicating the progress and status of the data processing and training stages.

---

\
After training the LSTM models for individual logical volumes, lvm_balancer generates key artifacts:

#### Data Plot:

-   **Location:** prediction/figures
-   **Purpose:** Visualize actual vs. predicted values for training and testing data.

#### Model File:

-   **Format:** .pt (e.g., PyTorch).
-   **Usage:** Serialized model for real-time predictions.

#### Scaler File:

-   **Format:** MinMaxScaler parameters.
-   **Usage:** Consistent data scaling in future predictions.
<div align="center">
  <img alt="Training Results: Plot of a portion of trained and tested data (actual and predicted)" src="https://github.com/hamza-hadj-aissa/lvm_balancer/blob/main/prediction/dummy/figures/model_7fedf800-2af3-40bd-b836-6393fe5e1241.png" width=900/>
  <p>Training Results: Plot of a portion of trained and tested data (actual and predicted)</p>
</div>

### Database Schema

Overview
The lvm_balancer system relies on an SQL database to store and manage the status and metrics of logical volumes. Understanding the database schema is essential for ensuring efficient data storage, retrieval, and analysis within the system.

#### Schema Description

Below is a high-level overview of the database schema:

<div align="center">
  <img alt="Database schema" src="https://github.com/hamza-hadj-aissa/lvm_balancer/blob/main/database/schema.png" width=900/>
  <p>Database schema</p>
</div>
## Usage

Run the main script to start scraping LVM statistics and storing them in the database:

## Supported Filesystems

The `lvm_balancer` system supports the following filesystems for logical volumes:

-   ext2
-   ext3
-   ext4
-   vfat
-   btrfs
-   f2fs
-   xfs

When using the system, ensure that your logical volumes are formatted with one of the listed filesystems to guarantee compatibility and accurate storage management.

## Workflow

The lvm_balancer system operates through a structured workflow to ensure efficient storage management. Once a model is trained for each logical volume using LSTM Recurrent Networks, the system begins scraping information from the logical volumes within the system at regular intervals. This data collection occurs every 10 minutes, allowing the system to continually monitor and analyze storage usage patterns.

After collecting data for six consecutive intervals (equivalent to one hour), the system leverages the trained models to perform optimized allocations for each logical volume. This proactive approach ensures timely adjustments and resource allocations, aligning with the predicted storage demands and enhancing overall system performance.

## Logs

Logs are categorized into three types:

-   Postgres (database-related)
-   LVM (LVM related commands)
-   Ansible (Ansible playbooks)
-   Main (main script logs)

You can find log messages in the console and adjust logging settings in the logs/Logger.py file.
