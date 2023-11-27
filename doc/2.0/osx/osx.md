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

two-way TSL：

1）方式一：使用keystore密码箱存储私钥、证书、信任证书方式

####  生成client和server端的秘钥keystore文件、证书文件、信任证书链，具体命令步骤如下：

 1. 创建一个包含服务器公钥和私钥的密钥库，并为其指定了一些属性：

    ```
    keytool -v -genkeypair -dname "CN=OSX,OU=Fate,O=WB,C=CN" -keystore server/identity.jks -storepass 123456 -keypass 123456 -keyalg RSA -keysize 2048 -alias server -validity 3650 -deststoretype pkcs12 -ext KeyUsage=digitalSignature,dataEncipherment,keyEncipherment,keyAgreement -ext ExtendedKeyUsage=serverAuth,clientAuth -ext SubjectAlternativeName:c=DNS:localhost,DNS:osx.local,IP:127.0.0.1
    ```

    - `keytool`: Java密钥和证书管理工具。
    - `-v`: 详细输出。
    - `-genkeypair`: 生成密钥对。
    - `-dname "CN=OSX,OU=Fate,O=WB,C=CN"`: 设置证书主题（Distinguished Name）。
    - `-keystore server/identity.jks`: 设置密钥库的文件路径和名称。
    - `-storepass 123456`: 设置密钥库的密码。
    - `-keypass 123456`: 设置生成的密钥对的密码。
    - `-keyalg RSA`: 使用RSA算法生成密钥对。
    - `-keysize 2048`: 设置密钥的大小为2048位。
    - `-alias server`: 设置密钥对的别名。
    - `-validity 3650`: 设置证书的有效期为3650天。
    - `-deststoretype pkcs12`: 指定密钥库的类型为PKCS12。
    - `-ext KeyUsage=digitalSignature,dataEncipherment,keyEncipherment,keyAgreement`: 扩展密钥用途。
    - `-ext ExtendedKeyUsage=serverAuth,clientAuth`: 扩展密钥用途。
    - `-ext SubjectAlternativeName:c=DNS:localhost,DNS:osx.local,IP:127.0.0.1`: 设置主体备用名称，包括DNS和IP地址。(这些主体备用名称的添加允许证书在与这些名称关联的主机或IP地址上使用，而不仅限于使用通用名称（Common Name，CN）字段中指定的主机名。这对于在服务器证书中包含多个主机名或IP地址是很有用的，特别是在使用SSL/TLS进行多主机名（SAN）认证时。)

    此命令用于生成包含服务器证书的密钥库，以便用于安全连接。请确保根据实际需求和环境进行适当的调整。

 2. 密钥库中导出证书，并将其保存为`.cer`文件:

    ```
    keytool -v -exportcert -file server/server.cer -alias server -keystore server/identity.jks -storepass 123456 -rfc
    ```

    - `keytool`: Java密钥和证书管理工具。
    - `-v`: 详细输出。
    - `-exportcert`: 导出证书。
    - `-file server/server.cer`: 指定导出证书的文件路径和名称。
    - `-alias server`: 指定要导出的证书条目的别名。
    - `-keystore server/identity.jks`: 指定密钥库的路径和名称。
    - `-storepass 123456`: 密钥库的密码。
    - `-rfc`: 以RFC 1421格式（Base64编码）输出证书。

    此命令用于从密钥库中导出服务器证书，并将其保存为`.cer`文件。请确保提供正确的密钥库路径、别名和密码，并根据需要更改导出证书的文件路径和名称。

 3.  从证书文件导入证书并将其添加到客户端的信任存储区:

    - ```
      keytool -v -importcert -file server/server.cer -alias server -keystore client/truststore.jks -storepass 123456 -noprompt
      ```

      `keytool`: Java密钥和证书管理工具。

    - `-v`: 详细输出。

    - `-importcert`: 导入证书。

    - `-file server/server.cer`: 指定要导入的证书文件路径和名称。

    - `-alias server`: 指定将证书存储在信任存储区时使用的别名。

    - `-keystore client/truststore.jks`: 指定信任存储区的路径和名称。

    - `-storepass 123456`: 信任存储区的密码。

    - `-noprompt`: 在导入证书时不提示用户确认。

    此命令用于将服务器证书导入到客户端的信任存储区，以建立与服务器的安全连接。确保提供正确的证书文件路径、别名、信任存储区路径和密码，并根据需要更改相关参数。 `-noprompt` 标志确保在导入证书时不需要手动确认。

 4. 生成客户端的密钥对和自签名证书:

    ```
    keytool -v -genkeypair -dname "CN=Suleyman,OU=Altindag,O=Altindag,C=NL" -keystore client/identity.jks -storepass 123456 -keypass 123456 -keyalg RSA -keysize 2048 -alias client -validity 3650 -deststoretype pkcs12 -ext KeyUsage=digitalSignature,dataEncipherment,keyEncipherment,keyAgreement -ext ExtendedKeyUsage=serverAuth,clientAuth
    ```

    - `keytool`: Java密钥和证书管理工具。
    - `-v`: 详细输出。
    - `-genkeypair`: 生成密钥对。
    - `-dname "CN=Suleyman,OU=Altindag,O=Altindag,C=NL"`: 设置证书主题（Distinguished Name，DN）。
    - `-keystore client/identity.jks`: 设置密钥库的文件路径和名称。
    - `-storepass 123456`: 设置密钥库的密码。
    - `-keypass 123456`: 设置生成的密钥对的密码。
    - `-keyalg RSA`: 使用RSA算法生成密钥对。
    - `-keysize 2048`: 设置密钥的大小为2048位。
    - `-alias client`: 设置密钥对的别名。
    - `-validity 3650`: 设置证书的有效期为3650天。
    - `-deststoretype pkcs12`: 指定密钥库的类型为PKCS12。
    - `-ext KeyUsage=digitalSignature,dataEncipherment,keyEncipherment,keyAgreement`: 扩展密钥用途。
    - `-ext ExtendedKeyUsage=serverAuth,clientAuth`: 扩展密钥用途。

    此命令用于生成包含客户端证书的密钥库，以用于安全连接。请确保提供正确的密钥库路径、别名和密码，并根据需要更改其他参数。

 5. 从客户端的密钥库中导出证书，并将其保存为`.cer`文件:

    ```
    keytool -v -exportcert -file client/client.cer -alias client -keystore client/identity.jks -storepass 123456 -rfc
    ```

    - `keytool`: Java密钥和证书管理工具。
    - `-v`: 详细输出。
    - `-exportcert`: 导出证书。
    - `-file client/client.cer`: 指定导出证书的文件路径和名称。
    - `-alias client`: 指定要导出的证书条目的别名。
    - `-keystore client/identity.jks`: 指定密钥库的路径和名称。
    - `-storepass 123456`: 密钥库的密码。
    - `-rfc`: 以RFC 1421格式（Base64编码）输出证书。

    此命令用于从客户端的密钥库中导出客户端证书，并将其保存为`.cer`文件。请确保提供正确的密钥库路径、别名和密码，并根据需要更改导出证书的文件路径和名称。

 6. 从客户端的证书文件中导入证书，并将其添加到服务器的信任存储区:

    ```
    keytool -v -importcert -file client/client.cer -alias client -keystore server/truststore.jks -storepass 123456 -noprompt
    ```
    
    - `keytool`: Java密钥和证书管理工具。
    - `-v`: 详细输出。
    - `-importcert`: 导入证书。
    - `-file client/client.cer`: 指定要导入的证书文件路径和名称。
    - `-alias client`: 指定将证书存储在信任存储区时使用的别名。
    - `-keystore server/truststore.jks`: 指定信任存储区的路径和名称。
    - `-storepass 123456`: 信任存储区的密码。
    - `-noprompt`: 在导入证书时不提示用户确认。
    
    此命令用于将客户端证书导入到服务器的信任存储区，以建立与客户端的安全连接。确保提供正确的证书文件路径、别名、信任存储区路径和密码，并根据需要更改相关参数。 `-noprompt` 标志确保在导入证书时不需要手动确认。

#### 完成以上步骤您将生成如下证书：

​			server文件夹包含： identity.jks 、server.cer、truststore.jks。

​			client文件夹包含： identity.jks 、client.cer、truststore.jks。

#### OSX相关配置：

- grpcs：

  broker.property配置（使用keystore方式，即方式1）：

  ```
  # 打开grpcs server开关
  open.grpc.tls.server= true
  # 是否使用keystore方式
  open.grpc.tls.use.keystore= true
  
  
  
  
  ```

  相关client端路由表配置：

  ```
  
  ```

  

- https：

  broker.property配置（使用keystore方式，即方式1）：

  ```
  
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

















