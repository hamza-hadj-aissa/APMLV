# APMLV: Automated Prediction and Management of Logical Volumes

`APMLV` is a project that leverages deep learning and automation to optimize the management of logical volumes resources. It consists of the following components:

-   A data extraction module that collects logical volume usage data from various sources and stores them in a SQL database
-   A data analysis and preprocessing module that cleans and transforms the data using Python
-   A machine learning module that trains an LSTM model for time series forecasting and makes predictions on future usage of logical volumes
-   An automation module that uses Ansible to adjust the logical volume configuration based on the predictions and improve the performance and efficiency of the system

## Table of Contents

-   [Project Prerequisites](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#prerequisites)
-   [Installation](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#installation)
-   [Configuration](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#configuration)
    -   [Ansible configuration](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#ansible-configuration)
-   [Project overview](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#project-overview)
    -   [Automated Prediction and Management of Logical Volumes using LSTM Recurrent Networks](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#automated-prediction-and-management-of-logical-volumes-using-lstm-recurrent-networks)
    -   [Training the model](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#training-the-model)
    -   [Model usage](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#model-usage)
    -   [Database schema](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#database-schema)
    -   [Supported filesystems](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#supported-filesystems)
-   [Workflow](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#workflow)
-   [Logging](https://github.com/hamza-hadj-aissa/APMLV/tree/main?tab=readme-ov-file#logging)

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
git clone https://github.com/hamza-hadj-aissa/APMLV.git
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

DB_NAME = "APMLV"
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

In the context of `APMLV`, the prediction of logical volumes (LV) usage is crucial for proactive management and optimization of storage resources. Traditional methods might fall short in capturing the dynamic and non-linear patterns of storage usage. To address this, we employ LSTM (Long Short-Term Memory) Recurrent Networks, a type of deep learning architecture known for its capability to model sequences and time-series data.

#### How LSTM Recurrent Networks Work

LSTM networks are a special kind of Recurrent Neural Network (RNN) that can learn and remember patterns over long sequences, making them particularly suitable for time-series data like storage usage metrics. Here's a simplified breakdown:

-   **Long-term Memory Cells:** LSTMs have memory cells that can maintain information over long sequences. This ability helps capture trends and patterns in storage usage data that traditional methods might overlook.

-   **Gates:** LSTMs have mechanisms called gates (input, forget, and output gates) that regulate the flow of information into and out of the memory cells. This gating mechanism allows LSTMs to decide what information to keep or discard, making them adept at handling sequences with long-range dependencies.

-   **Training:** The LSTM network is trained using historical storage usage data. During training, the network learns the underlying patterns and correlations in the data, enabling it to make accurate predictions.

#### Benefits

-   **Accurate Predictions:** LSTMs can capture intricate patterns and non-linear relationships in storage usage data, leading to more accurate predictions compared to traditional methods.

-   **Proactive Management:** By forecasting future storage usage, administrators can proactively allocate resources, prevent potential bottlenecks, and optimize the overall storage infrastructure.

-   **Scalability:** The LSTM model can be scaled to handle larger datasets and adapt to evolving storage usage patterns. (_It's essential to approach scaling with caution to ensure proper handling of the model_ ). Related article: [Incremental Ensemble LSTM Model towards Time Series Data](https://www.sciencedirect.com/science/article/abs/pii/S0045790621001592)


### Training the model

Each logical volume is associated with its own trained LSTM model, enabling tailored predictions and proactive storage management. This individualized approach allows `APMLV` to account for unique usage patterns, trends, and requirements specific to each logical volume

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
After training the LSTM models for individual logical volumes, `APMLV` generates key artifacts:

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
  <img alt="Training Results: Plot of a portion of trained and tested data (actual and predicted)" src="https://github.com/hamza-hadj-aissa/APMLV/blob/main/prediction/dummy/figures/model_7fedf800-2af3-40bd-b836-6393fe5e1241.png" width=900/>
  <p>Training Results: Plot of a portion of trained and tested data (actual and predicted)</p>
</div>


### Model usage

The LSTM-based prediction model in `APMLV` leverages historical LV usage metrics to forecast future usage trends. By analyzing past data points, the model can provide insights into potential storage demands, enabling more informed decision-making and efficient resources allocation.

#### Decision-making Process

The allocation of logical volumes is determined through a multi-step process:

-   **Predicting Future Usage:** Utilizing LSTM Recurrent Networks, the system forecasts the future usage of each logical volume. This prediction serves as a baseline for understanding the expected growth or reduction in storage requirements.

-   **Historical Proportion Analysis:** To maintain balance and fairness within the volume group, the historical proportion of each logical volume's usage relative to its neighbors is calculated. This analysis ensures that no single logical volume disproportionately consumes resources, leading to potential bottlenecks or inefficiencies.<br/>
        The mean proportion for a logical volume over a period of time T is
determined by the formula:
<div align="center">
    <img  src="https://latex.codecogs.com/svg.image?\large&space;\bg{red}Mean&space;Proportion=\frac{1}{n}\times\sum_{t=1}^{T}\frac{U_t}{\sum_{i=1}^{n}U_i_t}" title="Mean Proportion=\frac{1}{n}\times\sum_{t=1}^{T}\frac{U_t}{\sum_{i=1}^{n}U_i_t}" />
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where : <br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;n: The number of logical volumes in the volume group.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;U_t: The usage volume of a specific logical volume at time t.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;U_i_t: The usage of the ith logical volume in the volume group at time t.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;T: The period of time during which the usage of a logical volume is collected and predictions are made based on that series of usages.<br/>


-   **Priority Factor Integration:** The priority factor is incorporated into the allocation factor calculation, assigning weights or rankings to logical volumes based on their priority levels. Logical volumes with higher priority factors receive preferential treatment in the allocation process.<br/>
The formula for the Mean Proportion-to-Priority Factor is:
<div align="center">
    <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{red}Priority&space;Factor=\frac{Mean&space;Proportion}{{Priority}\times\frac{1}{Count(Priority)}}" title="Priority Factor=\frac{Mean Proportion}{{Priority}\times\frac{1}{Count(Priority)}}" />
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where :<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Priority : Numerical value representing the priority level of a logical volume.<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Count(Priority) : Function that counts the number of logical volumes with the same priority level.<br/>


&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;To scale the Mean Proportion-to-Priority Factor pi to ensure they sum up to 1, the scaled factor bi can be computed as:
<div align="center">
    <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{red}S_i=\frac{L_i}{\sum_{j=1}^{n}L_j}" title="S_i=\frac{L_i}{\sum_{j=1}^{n}L_j}" />
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where :<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;n : The number of logical volumes in the volume group<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;L : A list of Mean Proportion-to-Priority Factors [p1,p2,...,pn]<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;S : Scaled Mean Proportion-to-Priority Factors [b1,b2,...,bn]<br/>
-   **Demand-to-Space Ratio:** The calculation of the allocation factor is designed to ensure that the cumulative demands of all logical volumes are appropriately scaled relative to the available free space in the volume group. Thereby facilitating a balanced and efficient allocation strategy.<br/>
To determine the free allocation space in the volume group, we calculate Allocation/Reclaim size for each logical volume by :
<div align="center">
    <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{red}\begin{matrix}m_i\times&space;d_i&plus;p_i\\p_i\times&space;d_i&plus;p_i\end{matrix}\left\{\begin{matrix}d_i\geq&space;0\\d_i\;<0\end{matrix}\right." title="\begin{matrix}m_i\times d_i+p_i\\p_i\times d_i+p_i\end{matrix}\left\{\begin{matrix}d_i\geq 0\\d_i\;<0\end{matrix}\right." />
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;where :<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;pi : 20% of the model’s prediction ( i.e 20% of free space )<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;di : The difference between the prediction and the current filesystem size for each logical volume<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;mi : The scaled Mean Proportion-to-Priority Factor of the logical volume<br/>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Then, we sum up the results :<br/>
<div align="center">
    <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{red}Total&space;Allocations/Reclaims=\sum_{i=0}^{n}A_i&space;" title="Total Allocations/Reclaims=\sum_{i=0}^{n}A_i " />
</div>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Finally, The allocation factor is calculated by:<br/>
<div align="center">
    <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{red}Allocation&space;Factor=\frac{Volume&space;group&space;free&space;space}{Total&space;Allocation/Reclaim}" title="Allocation Factor=\frac{Volume group free space}{Total Allocation/Reclaim}" />
</div>

Finally, the allocation/reclaim of a logical volume is determined by comparing the allocation/reclaim size and the allocation/reclaim size scaled by the allocation factor, which are calculated as below:
*    *    **Allocation/Reclaim Size Scaled by the Allocation Factor:** 
  This metric introduces an additional layer of complexity by incorporating the allocation_factor. It can either magnify or reduce the initial allocation/reclaim size based on the value of the allocation_factor. 
 It is calculated as below:<br/>
 *    *    *    For a negative allocation/reclaim size :
<div align="center">
    <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{red}Adjustment=Max(AR,ARSAF)" title="Adjustment=Max(AR,ARSAF)" />
</div>

 *    *    *    For a positive allocation/reclaim size :
<div align="center">
    <img src="https://latex.codecogs.com/svg.image?\large&space;\bg{red}Adjustment=Min(AR,ARSAF)" title="Adjustment=Min(AR,ARSAF)" />
</div>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Where :<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AR : Represents for Allocation/Reclaim size<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;ARSAF : Represents for Allocation/Reclaim size scaled by the allocation factor<br/>

### Database Schema

#### Overview

The `APMLV` system relies on an SQL database to store and manage the status and metrics of logical volumes. Understanding the database schema is essential for ensuring efficient data storage, retrieval, and analysis within the system.

#### Schema Description

Below is a high-level overview of the database schema:

<div align="center">
  <img alt="Database schema" src="https://github.com/hamza-hadj-aissa/APMLV/blob/main/database/schema.png" width=900/>
  <p>Database schema</p>
</div>

## Supported Filesystems

The `APMLV` system supports the following filesystems for logical volumes:

-   ext2
-   ext3
-   ext4
-   vfat
-   btrfs (experimental)
-   f2fs
-   xfs (experimental)

When using the system, ensure that your logical volumes are formatted with one of the listed filesystems to guarantee compatibility and accurate storage management.

## Workflow

The `APMLV` system operates through a structured workflow to ensure efficient storage management. Once a model is trained for each logical volume using LSTM Recurrent Networks, the process of predictive storage managments begins for each volume group on each host:

-   1 - **Data Collection:** The system gathers data from the volume groups at consistent intervals.
-   2 - **Data Processing:** Collected data undergoes analysis to ascertain storage patterns and usage trends.
-   3 - **Threshold Monitoring:** Immediate adjustments are triggered if any logical volume within a group exceeds its usage threshold within a 6 \* 10-minute timeframe.
-   4 - **Predictive Analysis:** Using LSTM Recurrent Networks, predictive models are applied to anticipate future volume requirements.
-   5 - **Allocation Volume Computation:** The system computes the allocation volume for each logical volume, considering predicted usage, historical proportion, priority factor, and demand-to-space ratio.
-   6 - **Adjustments Execution:** Once adjustments are made, the time tracking resets, nitiating another monitoring cycle lasting 6 \* 10 minutes.

## Logs

Logs are categorized into three types:

-   Postgres (database-related)
-   LVM (LVM related commands)
-   Ansible (Ansible playbooks)
-   Main (Main script logs)

## Future improvements:
*    Multi-threaded Processing:
     +    **Description:** Implementing threads to assign each volume group to a separate thread can significantly speed up the processing time by allowing parallel execution of tasks.
     +    **Benefits:** Improved efficiency and reduced processing time
*    **Volume group size adjustments:**
     +    **Description:** Expanding the model to incorporate adjustments for volume group sizes, enabling dynamic resizing based on workload requirements and resource availability.
     +    **Benefits:** Greater flexibility and adaptability to changing system demands.
*    **Incremental Learning for LSTM Networks:**
     +    **Description:** Implementing incremental learning techniques for LSTM (Long Short-Term Memory) networks, allowing the model to continuously learn from new data without retraining the entire model.
     +    **Benefits:** Improved model accuracy over time, adaptability to evolving usage patterns, and reduced computational overhead for frequent updates.
*    **User Interface for Monitoring & Visualization:**
     +    **Description:** Developing a user-friendly interface that provides real-time monitoring and visualizations of logical volumes, allocation/reclaim metrics, and historical trends.
     +    **Benefits:** Enhanced user experience, easier data interpretation, and quicker decision-making capabilities for administrators and users.
