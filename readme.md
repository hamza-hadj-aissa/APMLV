# LVM Balancer

The project aims to develop a system that balances logical volumes. The status of the logical volumes on the machine will be extracted at regular intervals and saved into an SQL database. This data will then be analyzed using machine learning to predict future usage of each logical volume. Based on these predictions, the system will adjust the sizes of the logical volumes accordingly to optimize usage and performance.


## Installation

**1** -  Clone this repository to your machine by running

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
