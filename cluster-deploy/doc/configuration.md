[toc]

# 1. Overall
This document consists of 3 parts:

a. Module configurations;

b. Service management scripts;

c. Deployment scripts.

# 2. Module Configurations

## 2.1. Federation
Federation module handles task data communication (i.e. 'federation') among Federated Learning parties for own party.
### 2.1.1. applicationContext-federation.xml
No modification is required.
### 2.1.2. log4j2.properties
No modification is required.
### 2.1.3. federation.properties
Item              | Meaning             | Example / Value
------------------|---------------------|------------------------
party.id          |party id of FL participant      | e.g. 10000
service.port      |port to listen on    | federation defaults to 9394
meta.service.ip   |meta-service ip  | e.g. 172.16.153.xx
meta.service.port |meta-service port | defaults to 8590

## 2.2. Meta-Service
Meta-Service module stores metadata required by this arch.
### 2.2.1. applicationContext-meta-service.xml
No modification is required.
### 2.2.2. log4j2.properties
No modification is required.
### 2.2.3. meta-service.properties
Item              | Meaning             | Example / Value
------------------|---------------------|------------------------
party.id          |party id of FL participant | e.g. 10000
service.port      |port to listen on    | meta-service defaults to 8590
### 2.2.4. jdbc.properties
Item                   | Meaning             | Example / Value
-----------------------|---------------------|------------------------
jdbc.driver.classname  |jdbc driver's classname | recommendation: com.mysql.cj.jdbc.Driver
jdbc.url               |jdbc connection url  | modify as needed
jdbc.username          |database username    | modify as needed
jdbc.password          |database password    | modify as needed
target.project         |target project. Required by mybatis-generator    | fixed to meta-service
### 2.2.5. Database Configurations
#### 2.2.5.1. Database and Tables Creation
Please run the following SQL in this project:
arch/eggroll/meta-service/src/main/resources/create-meta-service.sql

#### 2.2.5.2. Adding Node Infomation
To deploy FATE in a distributed environment (i.e. cluster deploy), following modules are minimum for **1** party:

Module | Minimum requirement | Comments
-------|---------------------|------------
Roll   | exactly 1           | Advanced deployment in the next version
Egg (processor) | at least 1 | This will change in the next version
Egg (storage-service) | at least 1 | 
Federation | exactly 1       | 
Proxy  | at least 1          | 
Exchange | Inter-party communication can use any amount of exchange, included 0 (i.e. direct connection)

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

**d. Federation**  

No database record insertion is need for Federation module at this stage.

**e. Proxy**

For each Proxy, a database record should be inserted:
```
INSERT INTO node (ip, port, type, status) values 
('${proxy_ip}', '${proxy_port}', 'PROXY', 'HEALTHY')
```

## 2.3. Processor
Processor is used to execute user-defined functions.
### 2.3.1. service.sh
Modify variables in processor.sh base on your environment. 

Item                   | Meaning             | Example / Value
-----------------------|---------------------|------------------------
PORT                   |port to listen on    | processor defaults to 7888
DATADIR                |data storage dir     | must be the same with data dir in storage-service


## 2.4. Proxy (Shared with Exchange For Now)
Proxy (Exchange) is communication channel among parties.
### 2.4.1. applicationContext-proxy.xml
No modification is required.
### 2.4.2. log4j2.properties
No modification is required.
### 2.4.3. proxy.properties
Item              | Meaning                | Example / Value
------------------|------------------------|------------------------
coordinator       | same as party id       | e.g. 10000
ip                | ip to bind (in multi-interface env) | optional
port              | port to listen on      | proxy (exchange) defaults to 9370
route.table       | path to route table    | modify as needed
server.crt        | server certification path | only necessary in secure communication
server.key        | server private key path | only necessary in secure communication
root.crt          | path to certification of root ca | 暂时不填

### 2.4.4. route_table.json

Item              | Meaning                       | Example / Value
------------------|------------------------------|------------------------
default           | ip and port of exchange or default proxy | 172.16.153.xx / 9370
${partyId}        | federation ip and port of own party  | 172.16.153.yy / 9394

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


## 2.5. Roll
Roll module is responsible for accepting distributed job submission, job / data schedule and result aggregations.

### 2.5.1. applicationContext-roll.xml
No modification is required.
### 2.5.2. log4j2.properties
No modification is required.
### 2.5.3. roll.properties
Item              | Meaning             | Example / Value
------------------|---------------------|------------------------
party.id          |party id of FL participant | e.g. 10000
service.port      |port to listen on.   | roll defaults to 8011
meta.service.ip   |meta-service ip      | e.g. 172.16.153.xx
meta.service.port |meta-service port    | defaults to 8590

## 2.6. Storage-Service
Storage-Service module handles data storage on that single node.
### 2.6.1. log4j2.properties
No modification is required.
### 2.6.2. service.sh
No modification is required. But there are 2 command line mandatory arguments:

Item                   | Meaning             | Example / Value
-----------------------|---------------------|------------------------
PORT                   | port to listen on   | storage-service defaults to 7778
DATADIR                |data dir           | must be the same with processor's data dir

## 2.7. Serving-Server
Serving-Server is a online service for serving federated learning models.
### 2.7.1. log4j2.properties
No modification is required.
### 2.7.2. serving-server.properties
Item                            | Meaning                                                    | Example / Value
--------------------------------|------------------------------------------------------------|------------------------
ip                              |ip to bind                                                  | e.g. 127.0.0.1 
port                            |port to listen on                                           | e.g. 8001
workMode                        |working mode of serving, 0 for standalone and 1 for cluster | e.g. 1
party.id                        |party id of FL participant                                  | e.g. 10000
proxy                           |proxy address                                               | e.g. 172.16.153.xx:9370
roll                            |roll address                                                | e.g. 172.16.153.xx:8011
standaloneStoragePath           |standalone storage path if you deploy as standalone         | e.g. ${deployDir}/python/data/
modelCacheAccessTTL(hour)       |the model cache will be cleaned up after this time          | e.g. 12
modelCacheMaxSize               |maximum number of cached models                             | e.g. 50
OnlineDataAccessAdapter         |get host data by this adapter, it is a java class           | e.g. TestFile
InferenceResultProcessingAdapter|some processing can be done after the model results         | e.g. PassProcessing
### 2.7.3 best practice
- Use different proxy online service and offline training.
- Deploy at least two online services and use Load Balancer.

## 2.8. Task-Manager
Task Manager is a service for managing tasks. It can be used to start training tasks, upload and download data, publish models to serving, etc.
### 2.8.1. settings.py
Item         | Meaning                                                    | Example / Value
-------------|------------------------------------------------------------|------------------------
IP           |ip to bind                                                  | e.g. 127.0.0.1 
HTTP_PORT    |http port to listen on                                      | e.g. 6380
GRPC_PORT    |grpc port to listen on                                      | e.g. 6360
PARTY_ID     |party id of FL participant                                  | e.g. 10000
WORK_MODE    |working mode of serving, 0 for standalone and 1 for cluster | e.g. 1
DATABASE     |database settings                                           | 


## 2.8. API
APIs are interfaces exposed by the whole running architecture. Algorithm engineers / scientists can utilize FATE framework via API.
### 2.8.1 arch/conf/server_conf.json
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
        "172.153.16.1:8000",
        "172.153.16.2:8000"
    ]
  }
}
```

# 3. Service Management Scripts
## 3.1. Single Service Management - service.sh
```
usage: sh service.sh {start|stop|status|restart}
```
Arg. Seq | usage          | Meaning             
---------|----------------|---------------------
1        | start          | to start the service
1        | stop           | to stop the service
1        | status         | to check the service status
1        | restart        | to restart the service


## 3.2. Service Management in Top Level Dir - services.sh
```
usage: sh services.sh {all|current|[module1 module2 ...]} {start|stop|status|restart}
```
Arg. Seq | usage                           | Meaning             
---------|---------------------------------|---------------------
1        | all                             | ALL services
1        | current                         | CURRENT running services
1        | [module1 ... ]                  | one or more modules, separated by space
2        | start / stop / status / restart | 与3.1.定义一致

# 4. Deployment Scripts
## 4.1. Deploy Whole FATE Project - fate-deploy.sh
```
usage: sh fate-deploy.sh {cluster_name} {deploy|try|rollback|overwrite|restart}
```
Arg. Seq | usage      | Meaning             
---------|------------|--------------------
1        | eg. test / production | cluster name
2        | deploy     | REAL deploy. This operation utilize rsync to deploy incremental part of the project. But only jars / python sources will be deployed, configuration files and logs are not included. Old version will be back up to fate-deploy.old
2        | try        | TRY to deploy (i.e. dry run). No file change will occured. Users should run this before real deploy.
2        | rollback   | Roll back to last version stored in fate-deploy.old. Only rolls back jars and python sources. Configuration fils and logs are not included.
2        | overwrite  | OVERWRITE all files, including jars, python sources and configuration files. Usually used in first deploy.
2        | restart    | run "sh services.sh current restart" in remote. i.e. restarting current running services. See section 3.2.



## 4.2. Deploy Files/Directories - file-deploy.sh
```
usage: sh file-deploy.sh {cluster_name} {local_path} {remote_path}
```
Arg. Seq | usage             | Meaning             
---------|-------------------|---------------------
1        | test / production / {ip} | cluster name or single ip address
2        | local_path        | local path. can be a file or a dir
3        | remote_path       | remote path
