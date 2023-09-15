### Installing FATE Environment via Pypi Packages
#### 1. Installing Dependencies
- Prepare and install [conda](https://docs.conda.io/projects/miniconda/en/latest/) environment.
- Create a virtual environment:
```shell
# FATE runs on Python >= 3.8
conda create -n fate_env python=3.8
conda activate fate_env
```
- Install `fate_client`、`fate_flow` and `fate`:
```shell
pip install fate_client[fate,fate_flow]==2.0.0.b0
```

#### 2. Service Initialization
```shell
fate_flow init --ip 127.0.0.1 --port 9380 --home $HOME_DIR
```
- ip: The IP address where the service runs.
- port: The HTTP port for the service.
- home: The data storage directory, including data, models, logs, job configurations, sqlite.db, etc.

A successful initialization will return:
```shell
home: xxx
Init server completed!
```

#### 3. Starting the Service
The command to start the service is:
```shell
fate_flow start
```
You can use the command to check the service status:
```shell
fate_flow status
```

#### 4. Other Commands
- Stopping the Service
```shell
fate_flow stop
```

- Restarting the Service
```shell
fate_flow restart
```
- Viewing Version Information
```shell
fate_flow version
```