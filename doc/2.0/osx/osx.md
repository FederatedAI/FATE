# 一、背景：

FATE1.X维护了多套通信架构，包括eggroll、spark+pulsar+nginx 、spark+rabbitmq+nginx。

##### 下图为FATE1.X采用eggroll为计算引擎时的通信架构

![fate_old_eggroll](../images/fate_old_eggroll.png)

##### 下图为FATE1.X采用spark为计算引擎时的通信架构

![fate_old_spark](../images/fate_old_spark.png)

如上所示，FATE1.X通信架构有一些弊端如下：

- 需要维护多套通信组件来支持不同的计算引擎
- 多种计算引擎组网困难，难以统一路由以及流控
- eggroll通信只支持同步rpc调用+流式传输，不支持消息队列类型的异步传输
- pulsar 以及rabbitmq等集群不易安装以及维护，程序不容易感知集群间传输时出现的网络异常
- 不容易对消息队列组件进行自定义开发等

为了解决以上问题，我们预备在FATE2.X中使用统一的通信组件OSX，统一支持不同计算引擎

# 二、新组件OSX:



## 新组件特性：

- 传输接口兼容FATE1.X版本  FATE2.X版本

- 按照《金融业隐私计算互联互通平台技术规范》实现传输接口

- 支持多种计算引擎传输，包括eggroll、spark

- 传输模式支持rpc、消息队列

- 传输协议支持grpc（rpc、消息队列模式）、http1.1(消息队列模式)

- 支持作为exchange中心节点部署 ，支持FATE1.X  、FATE2.X 接入

- 路由配置与原eggroll基本一致，降低了移植难度

  

## 组件设计：



## ![osx_component](../images/osx_component.png)

## 部署架构：

采用eggroll作为计算引擎时的部署架构:

![osx_on_eggroll](../images/osx_on_eggroll.png)

上图为采用spark作为计算引擎时的部署架构:







![osx_on_spark.drawio](../images/osx_on_spark.drawio.png)



与其他厂商互联互通互联互通



![union](../images/union.png)



## 配置：

以下为osx最简配置,配置文件位于 {部署目录}/conf/broker/broker.properties

```properties
grpc.port= 9377   （服务监听的grpc端口）
# eg: 9999,10000,10001 （本方partyId, 对应于互联互通协议中的nodeId）
self.party=10000  
# （若使用eggroll作为计算引擎，此处填入eggroll cluster-manager 的ip）
eggroll.cluster.manager.ip = localhost
# （若使用eggroll作为计算引擎，此处填入eggroll cluster-manager 的端口）
eggroll.cluster.manager.port = 4670
```

全部配置：

| 名称                                           | 含义                                                         | 默认值               | 是否必须配置                        | 说明                                                         |
| ---------------------------------------------- | ------------------------------------------------------------ | -------------------- | ----------------------------------- | ------------------------------------------------------------ |
| grpc.port                                      | 服务监听grpc端口（非TLS）                                    | 9370                 | 否                                  | 该端口用做默认的集群内部通信端口 ，若是用于非生产环境，出于方便测试以及调试考虑，可以将此端口作为集群间通信端口。若是生产环境使用，出于安全考虑，不应该将此对外暴露，而是将使用TLS 的端口对外暴露 ，参考配置open.grpc.tls.server grpc.tls.port |
| self.party                                     | 本方partyId                                                  | 无                   | 是                                  | **此配置非常重要，需要在部署集群前确定己方partyId，此处的配置会影响请求是否能正确路由** |
| eggroll.cluster.manager.ip                     | 若使用eggroll作为计算引擎，此处填入eggroll cluster-manager 的ip | 无                   | 否                                  |                                                              |
| eggroll.cluster.manager.port                   | 若使用eggroll作为计算引擎，此处填入eggroll cluster-manager 的端口 | 无                   | 否                                  |                                                              |
| open.grpc.tls.server                           | 是否开启使用TLS的grpc端口                                    | false                | 否                                  | 开启之后，服务将会监听一个使用TLS的grpc端口                  |
| grpc.tls.port                                  | 服务监听grpc端口(使用TLS)                                    | 无                   | 若open.grpc.tls.server =true 则必填 | 出于安全考虑，在生产上一般将此端口用做对外通信。而通过grpc.port配置的端口，则用于集群内部组件之间的通信。 |
| open.http.server                               | 是否开启http1.x协议端口(非TLS)                               | false                | 否                                  | http协议目前只适用于队列模式传输，且FATE1.X版本接口不支持http协议，若使用了其他厂家提供的使用http协议的算法容器（FATE算法默认使用grpc），则可以开启httpServer，该配置默认关闭 |
| http.port                                      | httpServer端口(非TLS)                                        | 无                   | 若open.http.server =true 则必填     |                                                              |
| open.https.server                              | 是否开启http1.x协议端口(TLS)                                 | false                | 否                                  |                                                              |
| http.context.path                              | http服务端配置                                               | /v1                  | 否                                  | eg:  http://127.0.0.1:9370/v1/interconn/chan/invoke  中的v1字符串 |
| http.servlet.path                              | http服务端配置                                               | /*                   | 否                                  | eg:  http://127.0.0.1:9370/v1/interconn/chan/invoke   中v1/后的内容 |
| https.port                                     | httpServer端口(使用TLS)                                      | 无                   | 若open.https.server=true 则必填     |                                                              |
| bind.host                                      | 绑定本地ip（适用于http 与grpc server）                       | 0.0.0.0              | 否                                  |                                                              |
| grpc.server.max.concurrent.call.per.connection | 服务端单个grpc链接最大并发                                   | 1000                 | 否                                  |                                                              |
| grpc.server.max.inbound.message.size           | 服务端单个grpc包最大大小                                     | 2147483647           | 否                                  |                                                              |
| grpc.server.max.inbound.metadata.size          | 服务端单个grpc包最大 metadata 大小                           | 134217728            | 否                                  |                                                              |
| grpc.server.flow.control.window                | 服务端grpc流控窗口大小                                       | 134217728            | 否                                  |                                                              |
| grpc.server.keepalive.without.calls.enabled    | 服务端grpc是否允许连接没有调用是保持存活                     | true                 | 否                                  |                                                              |
| grpc.client.max.inbound.message.size           | 客户端单个grpc包最大大小                                     | 2147483647           | 否                                  |                                                              |
| grpc.client.flow.control.window                | 客户端grpc流控窗口大小                                       | 134217728            | 否                                  |                                                              |
|                                                |                                                              |                      |                                     |                                                              |
| queue.max.free.time                            | 队列最大空闲时间                                             | 43200000（单位毫秒） | 否                                  | 空闲时间超过该配置的队列，将会被回收，释放本地资源           |
| queue.check.interval                           | 检查队列空闲定时任务间隔                                     | 60000（单位毫秒）    | 否                                  |                                                              |
| consume.msg.waiting.timeout                    | 消费阻塞最大时间                                             | 3600000              | 否                                  | 若不同厂商算法组件消费接口中未指定超时时间，则使用配置作为超时时间 |
| grpc.oncompleted.wait.timeout                  | grpc流式传输中当一方已经完成传输后，等待另一方完成的时间     | 600（单位秒）        | 否                                  | grpc流式传输接口中使用                                       |



## 路由：

路由配置相关文件为{部署目录}/conf/broker/route_table.json ,与eggroll组件rollsite保持一致，下面介绍在不使用证书的情况下的操作步骤：

先检查{部署目录}/conf/broker/broker.properties 中配置 self.party,如下所示，则代表本方partyId为9999，（若是与遵循互联互通协议的其他厂商隐私计算产品对接，此处对应于互联互通协议的nodeId）

```
self.party=9999
```



**若发现该配置不符合预期，则需要修改成预期的partyId，并重启应用。**

本方partyId ：9999  （若是与遵循互联互通协议的其他厂商隐私计算产品对接，此处对应于互联互通协议的nodeId）

若对方partyId 为10000  （若是与遵循互联互通协议的其他厂商隐私计算产品对接，此处对应于互联互通协议的nodeId），则按照如下配置

```json
{
  "route_table":
  {
    "9999":      //己方partyId 9999  ，  
    {
      "fateflow":[    //配置己方路由只需要配置fateflow 地址就可以，需要注意这里需要配置fateflow的grpc端口，默认是9360
        {
          "port": 9360,   
          "ip": "localhost"
        }
      ]
    },
    "10000":{  //对方partyId 10000
      "default":[{   //配置对方路由，只需要配置default 地址就可以  , 地址为对方的osx grpc端口
        "port": 9370,
        "ip": "192.168.xx.xx"  
      }]

    }
  },
  "permission":
  {
    "default_allow": true
  }
}
```

**路由表修改之后不需要重启应用，系统会自动读取。**

## 

```protobuf

```



## 部署教程

1. 下载源码，打包机器需要安装好maven  + jdk
2. 进入源码目录/deploy,  执行sh auto-package.sh, 执行完之后会在当前目录出现osx.tar.gz。

## 部署：

1. 部署机器需要安装jdk1.8+
2. 解压osx.tar.gz 
3. 进入部署目录，执行sh service.sh start



### 日志分析：







### 证书相关：

#### OSX相关配置：

- grpcs：

  broker.property配置（使用keystore方式，即方式1）：

  ```
  # 打开grpcs server开关
  open.grpc.tls.server= true
  # 是否使用keystore方式（默认为false）
  open.grpc.tls.use.keystore= true
  #server端密码箱路径以及密码
  server.keystore.file=
  server.keystore.file.password=
  #server端信任证书keystore路径及密码
  server.trust.keystore.file=
  server.trust.keystore.file.password=
  
  ```

  相关client端路由表配置：

  ```
  "default": [
          {
            "protocol": "grpc",
            "keyStoreFile": "D:/webank/osx/test3/client/identity.jks",
            "keyStorePassword": "123456",
            "trustStoreFile": "D:/webank/osx/test3/client/truststore.jks",
            "trustStorePassword": "123456",
            "useSSL": true,
            "port": 9885,
            "ip": "127.0.0.1"
          }
        ]
  ```

  

- https：

  broker.property配置（使用keystore方式，即方式1）：

  ```
  # grpcs端口
  https.port=8092
  # 打开grpcs server开关
  open.https.server= true
  # server端密码箱路径以及密码
  server.keystore.file=
  server.keystore.file.password=
  # server端信任证书keystore路径及密码
  server.trust.keystore.file=
  server.trust.keystore.file.password=
  
  ```
  
  相关client端路由表配置：
  
  ```
  
  ```
  
  

2）方式二：单独文件存储私钥、证书、信任证书方式

​	生成命令：

​	

#### OSX相关配置：

- grpcs：

  broker.property配置（使用非keystore方式，即方式2）：

  ```
  # 打开grpcs server开关
  open.grpc.tls.server= true
  # 是否使用keystore方式
  open.grpc.tls.use.keystore= false
  
  
  
  
  ```

  相关client端路由表配置：

  ```
  
  ```

  

- https：

  broker.property配置（使用非keystore方式，即方式2）：

  ```
  
  ```

  相关client端路由表配置：

  ```
  
  ```

  



### 常见问题：

















