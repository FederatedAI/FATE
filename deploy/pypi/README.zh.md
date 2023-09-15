### Pypi包安装FATE环境
#### 1. 安装依赖
- [conda](https://docs.conda.io/projects/miniconda/en/latest/)环境准备及安装
- 创建虚拟环境
```shell
# fate的运行环境为python>=3.8
conda create -n fate_env python=3.8
conda activate fate_env
```
- 安装fate_client、fate_flow和fate
```shell
pip install fate_client[fate,fate_flow]==2.0.0.b0
```

#### 2. 服务初始化
```shell
fate_flow init --ip 127.0.0.1 --port 9380 --home $HOME_DIR
```
- ip: 服务运行ip
- port：服务运行时的http端口
- home: 数据存储目录。主要包括：数据/模型/日志/作业配置/sqlite.db等内容

初始化成功将返回：
```shell
home: xxx
Init server completed!
```

#### 3. 启动服务
启动服务命令为:
```shell
fate_flow start
```
你可以使用查询服务状态命令查看服务是否启动成功：
```shell
fate_flow status
```

#### 4. 其他命令
- 停止服务
```shell
fate_flow stop
```

- 重启服务
```shell
fate_flow restart
```
- 查看版本信息
```shell
fate_flow version
```
