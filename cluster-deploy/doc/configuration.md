[toc]

# 1. Overall
This document consists of 1 parts:

a. FATE Module configurations;

b.EggRoll Module configurations;

# 2. FATE Module Configurations

## 2.1. Proxy (Shared with Exchange For Now)

Proxy (Exchange) is communication channel among parties.

### 2.1.1. proxy/conf/applicationContext-proxy.xml

No modification is required.

### 2.1.2. proxy/conf/log4j2.properties

No modification is required.

### 2.1.3. proxy/conf/proxy.properties

| Item        | Meaning                             | Example / Value                        |
| ----------- | ----------------------------------- | -------------------------------------- |
| coordinator | same as party id                    | e.g. 10000                             |
| ip          | ip to bind (in multi-interface env) | optional,If empty, bind to 0.0.0.0     |
| port        | port to listen on                   | proxy (exchange) defaults to 9370      |
| route.table | path to route table                 | modify as needed                       |
| server.crt  | server certification path           | only necessary in secure communication |
| server.key  | server private key path             | only necessary in secure communication |
| root.crt    | path to certification of root ca    | null                                   |

### 2.1.4. proxy/conf/route_table.json

| Item       | Meaning                                  | Example / Value     |
| ---------- | ---------------------------------------- | ------------------- |
| default    | ip and port of exchange or default proxy | 192.168.0.xx / 9370 |
| ${partyId} | federation ip and port of own party      | 192.168.0.yy / 9394 |

example:

```
{
  "route_table": {
    "default": {
      "default": [
        {
          "ip": "127.0.0.1",
          "port": 9999
        }
      ]
    },
    "10000": {
      "default": [
        {
          "ip": "127.0.0.1",
          "port": 8889
        }
      ],
      "fateflow": [
        {
          "ip": "127.0.0.1",
          "port": 9360
        }
      ],
      "serving": [
        {
          "ip": "127.0.0.1",
          "port": 8001
        }
      ]
    },
    "9999": {
      "default": [
        {
          "ip": "127.0.0.1",
          "port": 8890
        }
      ],
      "manager": [
        {
          "ip": "127.0.0.1",
          "port": 9360
        }
      ],
      "serving": [
        {
          "ip": "127.0.0.1",
          "port": 8001
        }
      ]
    }
  },
  "permission": {
    "default_allow": true
  }
}
```

## 2.2. Federation

Federation module handles task data communication (i.e. 'federation') among Federated Learning parties for own party.
### 2.2.1. federation/conf/applicationContext-federation.xml
No modification is required.
### 2.2.2. federation/conf/log4j2.properties
No modification is required.
### 2.2.3. federation/conf/federation.properties
Item              | Meaning             | Example / Value
------------------|---------------------|------------------------
party.id          |party id of FL participant      | e.g. 10000
service.port      |port to listen on    | federation defaults to 9394
meta.service.ip   |meta-service ip  | e.g. 192.168.0.xx 
meta.service.port |meta-service port | defaults to 8590

## 2.3. python API

APIs are interfaces exposed by the whole running architecture. Algorithm engineers / scientists can utilize FATE framework via API.

### 2.3.1 python/arch/conf/server_conf.json

```
{
  "servers": {
    "roll": {
      "host": "localhost",  # ip address of roll module
      "port": 8011          # port of roll module
    },
    "federation": {
      "host": "localhost",  # ip address of federation module
      "port": 9394          # port of federation module
    },
    "manager": {             
      "host": "localhost", # ip address of task_manager
      "grpc.port": 9360,   # grpc port of task_manager
      "http.port": 9380    # http port of task_manager
    },
    "proxy": {
       "host": "localhost", # ip address of proxy module
       "port": 9370         # port address of proxy module
    },
    "servings": [
        "192.168.1:8000",
        "192.168.2:8000"
    ]
  }
}
```

## 2.4. FateFlow

fateflow is a service for managing tasks. It can be used to start training tasks, upload and download data, publish models to serving, etc.

### 2.4.1. python/fate_flow/settings.py

| Item      | Meaning                                                     | Example / Value |
| --------- | ----------------------------------------------------------- | --------------- |
| IP        | ip to bind                                                  | e.g. 127.0.0.1  |
| HTTP_PORT | http port to listen on                                      | e.g. 6380       |
| GRPC_PORT | grpc port to listen on                                      | e.g. 6360       |
| PARTY_ID  | party id of FL participant                                  | e.g. 10000      |
| WORK_MODE | working mode of serving, 0 for standalone and 1 for cluster | e.g. 1          |
| DATABASE  | database settings                                           |                 |
| REDIS     | redis settings                                              |                 |

## 2.5. FateBoard

 FATEBoard as a suite of visualization tool for federated learning modeling designed to deep explore models and understand models easily and effectively. 

[detailed configuration](https://github.com/FederatedAI/FATE-Board)

## 2.6. Serving-Server

Serving-Server is a online service for serving federated learning models.

[detailed configuration](https://github.com/FederatedAI/FATE-Serving)



# 3.EggRoll Module Configurations

## 3.1. Meta-Service

Meta-Service module stores metadata required by eggroll.

### 3.1.1. eggroll/meta-service/conf/applicationContext-meta-service.xml

No modification is required.

### 3.1.2. eggroll/meta-service/conf/log4j2.properties

No modification is required.

### 3.1.3. eggroll/meta-service/conf/meta-service.properties

| Item                  | Meaning                                       | Example / Value                          |
| --------------------- | --------------------------------------------- | ---------------------------------------- |
| party.id              | party id of FL participant                    | e.g. 10000                               |
| service.port          | port to listen on                             | meta-service defaults to 8590            |
| jdbc.driver.classname | jdbc driver's classname                       | recommendation: com.mysql.cj.jdbc.Driver |
| jdbc.url              | jdbc connection url                           | modify as needed                         |
| jdbc.username         | database username                             | modify as needed                         |
| jdbc.password         | database password                             | modify as needed                         |
| target.project        | target project. Required by mybatis-generator | fixed to meta-service                    |

### 3.1.4. Database Configurations

#### 3.1.4.1. Database and Tables Creation

Please run the following SQL in this project: framework/meta-service/src/main/resources/create-meta-service.sql

#### 3.1.4.2. Adding Node Infomation

To deploy Eggroll in a distributed environment (i.e. cluster deploy), following modules are minimum for **1** party:

| Module                | Minimum requirement | Comments |
| --------------------- | ------------------- | -------- |
| Roll                  | exactly 1           |          |
| Egg (processor)       | at least 1          |          |
| Egg (storage-service) | at least 1          |          |
| federation            | exactly 1           |          |
| Proxy                 | at least 1          |          |

**a. Roll**

For each Roll, a database record should be inserted:

```
INSERT INTO node (ip, port, type, status) values 
('${roll_ip}', '${roll_port}', 'ROLL', 'HEALTHY')
```

**b. Processor**

For each Processor, a database record should be inserted:

```
INSERT INTO node (ip, port, type, status) values 
('${processor_ip}', '${processor_port}', 'EGG', 'HEALTHY')
```

**c. Storage-Service**

For each Storage-Service, a database record should be inserted:

```
INSERT INTO node (ip, port, type, status) values 
('${storage_service_ip}', '${storage_service_port}', 'STORAGE', 'HEALTHY')
```

**d. federation**

No database record insertion is need for Clustercomm module at this stage.

**e. Proxy**

For each Proxy, a database record should be inserted:

```
INSERT INTO node (ip, port, type, status) values 
('${proxy_ip}', '${proxy_port}', 'PROXY', 'HEALTHY')
```

## 3.2. Roll

Roll module is responsible for accepting distributed job submission, job / data schedule and result aggregations.

### 3.2.1. eggroll/roll/conf/applicationContext-roll.xml

No modification is required.

### 3.2.2. eggroll/roll/conf/log4j2.properties

No modification is required.

### 3.2.3. eggroll/roll/conf/roll.properties

| Item                       | Meaning                    | Example / Value       |
| -------------------------- | -------------------------- | --------------------- |
| party.id                   | party id of FL participant | e.g. 10000            |
| service.port               | port to listen on.         | roll defaults to 8011 |
| meta.service.ip            | meta-service ip            | e.g. 127.0.0.xx       |
| meta.service.port          | meta-service port          | defaults to 8590      |
| eggroll.compatible.enabled | compatibility enabled      | true                  |

## 3.3. Storage-Service-cxx

Storage-Service-cxx module handles data storage on that single node.

### 3.3.1. eggroll/services.sh

No modification is required. But there are 2 command line mandatory arguments:

| Item    | Meaning           | Example / Value                            |
| ------- | ----------------- | ------------------------------------------ |
| PORT    | port to listen on | storage-service-cxx defaults to 7778       |
| DATADIR | data dir          | must be the same with processor's data dir |

## 3.4. API

APIs are interfaces exposed by the whole running architecture. Algorithm engineers / scientists can utilize Eggroll framework via API.

### 3.4.1 eggroll/python/eggroll/conf/server_conf.json

```
{
  "servers": {
    "roll": {
      "host": "localhost",  # ip address of roll module
      "port": 8011          # port of roll module
    },
    "clustercomm": {
      "host": "localhost",  # ip address of federation module
      "port": 9394          # port of federation module
    },
    "proxy": {
       "host": "localhost", # ip address of proxy module
       "port": 9370         # port address of proxy module
    }
  }
}
```

## 3.5. Egg

Egg used to execute user-defined functions.

### 3.5.1. eggroll/egg/conf/applicationContext-egg.xml

No modification is required.

### 3.5.2. eggroll/egg/conf/log4j2.properties

No modification is required.

### 3.5.3 eggroll/egg/conf/egg.properties

| Item                                          | Meaning                          | Example / Value                                 |
| --------------------------------------------- | -------------------------------- | ----------------------------------------------- |
| party.id                                      | party id of FL participant       | e.g. 10000                                      |
| service.port                                  | port to listen on.               | egg defaults to 7888                            |
| eggroll.computing.engine.names                | computing engine name of eggroll | defaults to processor                           |
| eggroll.computing.processor.bootstrap.script  | path of processor-starter.sh     | modify as needed                                |
| eggroll.computing.processor.start.port        | port of processor start          | defaults to 5000                                |
| eggroll.computing.processor.venv              | path of python virtualenv        | modify as needed                                |
| eggroll.computing.processor.engine-path       | path of processor.py             | modify as needed                                |
| eggroll.computing.processor.data-dir          | data dir                         | must be the same with storage's data dir        |
| eggroll.computing.processor.logs-dir          | log dir of processor             | modify as needed                                |
| eggroll.computing.processor.session.max.count | the count of processors          | modify as needed, but no more than server cores |
| eggroll.computing.processor.python-path       | path of python                   | User-defined python function library path       |
