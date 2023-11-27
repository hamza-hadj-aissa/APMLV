# LVM Balancer

The project aims to develop a system that balances logical volumes. The status of the logical volumes on the machine will be extracted at regular intervals and saved into an SQL database. This data will then be analyzed using machine learning to predict future usage of each logical volume. Based on these predictions, the system will adjust the sizes of the logical volumes accordingly to optimize usage and performance.

## Table of Contents

-   [Prerequisites](#prerequisites)
-   [Installation](#installation)
-   [Configuration](#configuration)
-   [Usage](#usage)

### Prerequisites

Before running the project, make sure the following software and dependencies are installed on your machine:

-   LVM (Logical Volume Manager)
-   Python 3.x
-   Pip
-   PostgreSQL

### Installation

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

### Configuration

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


### Usage
Run the main script to start scraping LVM statistics and storing them in the database:
```bash
python main.py
```
#### Logs
Logs are categorized into three types: 
- Postgres (database-related)
- LVM (LVM related commands),
- Main (main script logs).


You can find log messages in the console and adjust logging settings in the logs/Logger.py file.
