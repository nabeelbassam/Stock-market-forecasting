# Time-series forcasting project

## About Project
This project leverage XGBoost for stock price time-series analysis which can be considered an advancement of the traditional modelling techniques such as ARIMA. This approach helps in improving our results and speed of modelling by considering multiple features rather than one feature.
## Dataset

The data that I used has 1259  timestamps with the value of this stock on that day, the data can found from the [data](data/prices.txt) folder.


## How to Use

To use this project on your system, follow the following steps:

1. Clone this repository onto your system by typing the following command on your Command Prompt

2. Download all libraries:
    
   Using pip
   ```bash
   pip install -r requirements.txt
   ```
   Using Anaconda
   ```bash
   conda create --name env_name --file requirements.txt
   ```
3. Run forcast.py by typing the following command on your command line:
    ```bash
    python forcast.py [--data "data_path"][--ratio "training ratio"]
    ```
   **Notes:**
   * The default training ration is 0.8
   * The default data path is [prices.txt](data/prices.txt)  
   * You can find all the experiments in “notebooks/experiments.ipynb”
