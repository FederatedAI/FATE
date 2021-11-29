# FATE Exchange with Pulsar 部署指南
[English](fate-exchange_deployment_with_pulsar.md)

## 星型组网

使用pulsar作为传输服务可以支持星型部署，其中心节点是一个SNI(Server Name Indication)代理服务，具体的代理服务可使用[Apache Traffic Server](https://trafficserver.apache.org/)。具体的代理流程如下：
1. 客户端向代理服务器发送TLS Client Hello请求，请求中带有SNI字段，该字段声明了客户端想要连接的远端服务器的域名或主机名。
2. 代理服务器根据SNI的字段以及本身的路由信息与远端服务器建立TCP tunnel并转发客户端的TLS Hello。
3. 远端服务器发送TLS Server Hello给客户端并接着完成TLS握手。
4. TCP链接建立，客户端和远端服务器正常通讯。

### 具体部署方式
接下来将基于SNI代理的模式建立一个联邦学习网络，由于涉及到证书的生成，因此可以用一个统一的域名后缀来标识这个网络，如"fate.org"。这样网络中的各个实体可用`${party_id}.fate.org`的方式标识，如party 10000所使用证书的CN为"10000.fate.org"。

#### 规划
主机名 | IP地址 | 操作系统 | 安装软件 | 服务
-------|--------|----------|----------|-----
proxy.fate.org | 192.168.0.1 | CentOS 7.2/Ubuntu 16.04 | ats | ats
10000.fate.org | 192.168.0.2 | CentOS 7.2/Ubuntu 16.04 | pulsar | pulsar
9999.fate.org  | 192.168.0.3 | CentOS 7.2/Ubuntu 16.04 | pulsar | pulsar

具体架构图如下图所示，pulsar服务"10000.fate.org"属于ID为10000的组织，而pulsar服务"9999.fate.org"属于ID为9999的组织，"proxy.fate.org"为ats服务，是星型网络的中心。
<div style="text-align:center", align=center>
<img src="../../images/pulsar_sni_proxy.png" />
</div>

#### 证书生成
由于SNI代理基于TLS，因此需要为ATS和pulsar服务配置证书，首先生成的是CA证书，然后为ats和pulsar服务颁发"CN"与其主机名相同（在现实情况下证书的"CN"可与主机名不同）的证书。

##### 生成CA证书
输入下面的命令为 CA 创建一个目录，并将此 openssl 配置文件 放入该目录中。
``` bash
$ mkdir my-ca
$ cd my-ca
$ wget https://raw.githubusercontent.com/apache/pulsar/master/site2/website/static/examples/openssl.cnf
$ export CA_HOME=$(pwd)
```

输入下面的命令来创建必要的目录、密钥和证书。
``` bash
$ mkdir certs crl newcerts private
$ chmod 700 private/
$ openssl genrsa -aes256 -out private/ca.key.pem 4096
$ touch index.txt
$ echo 1000 > serial
$ chmod 400 private/ca.key.pem
$ openssl req -config openssl.cnf -key private/ca.key.pem \
    -new -x509 -days 7300 -sha256 -extensions v3_ca \
    -out certs/ca.cert.pem
$ chmod 444 certs/ca.cert.pem
```
在上面的命令中，生成密钥和证书需要进一步的交互，用户根据提示输入即可，对于不熟悉x509证书的用户来说一般除了`Common Name`之外其他都可使用默认值。一旦上述命令运行完毕，则CA相关的证书以及密钥均已生成。
其中：
- certs/ca.cert.pem 保存的是CA的证书文件
- private/ca.key.pem 保存是CA的密钥文件

##### 为10000.fate.org生成证书
1. 生成目录存储证书文件
```
$ mkdir 10000.fate.org
```

2. 输入下面的命令来生成密钥。
```
$ openssl genrsa -out 10000.fate.org/broker.key.pem 2048
```

3. Broker 需要密钥使用 PKCS 8 格式，因此输入以下命令进行转换。
```
$ openssl pkcs8 -topk8 -inform PEM -outform PEM \
      -in 10000.fate.org/broker.key.pem -out 10000.fate.org/broker.key-pk8.pem -nocrypt
```

4. 输入下面的命令生成证书请求，其中`Common Name`输入**10000.fate.org**
```
$ openssl req -config openssl.cnf \
    -key 10000.fate.org/broker.key.pem -new -sha256 -out 10000.fate.org/broker.csr.pem
```

5. 输入下面的命令获取证书颁发机构的签名。
```
$ openssl ca -config openssl.cnf -extensions server_cert \
    -days 1000 -notext -md sha256 \
    -in 10000.fate.org/broker.csr.pem -out 10000.fate.org/broker.cert.pem
```
此时，"10000.fate.org"目录下存放了证书"broker.cert.pem"，和一个密钥 "broker.key-pk8.pem"。此时客户端可以配合CA证书来对broker服务进行验证。

##### 为9999.fate.org生成证书
"9999.fate.org"证书的生成与上述的步骤一致，第4步的`Common Name`输入为**9999.fate.org**。

以下操作将默认"9999.fate.org"的证书已生成并放置在"9999.fate.org"目录下。

##### 为proxy.fate.org生成证书
"proxy.fate.org"证书的生成与上述的步骤一致，第3部的转化可省略，第5步的`Common Name`输入为**proxy.fate.org**。

以下操作将默认"proxy.fate.org"的证书已生成并放置在"proxy.fate.org"目录下，证书和私钥的分别为"proxy.cert.pem"，"proxy.key.pem"

#### 部署Apache Traffic Server
##### 安装Apache Traffic Server
1. 登录"proxy.fate.org"主机根据操作系统，按照此[文档](https://github.com/apache/trafficserver/tree/9.0.0)准备依赖软件。

2. 下载Apache Traffic server 9.0
```
$ wget https://apache.claz.org/trafficserver/trafficserver-9.0.0.tar.bz2
```

3. 解压并安装
```
$ mkdir /opt/ts
$ tar xf trafficserver-9.0.0.tar.bz2
$ cd trafficserver-9.0.0
$ ./configure --prefix /opt/ts
$ make                   
$ make install
$ echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/ts/lib' >> ~/.profile
$ source ~/.profile
```

当命令执行完毕，traffic server会被安装在`/opt/ts`目录下，配置文件的路径为`/opt/ts/etc/trafficserver/`。

##### 启动Apache Traffic Server服务
1. 修改ATS配置
- /opt/ts/etc/trafficserver/records.config
```
CONFIG proxy.config.http.cache.http INT 0
CONFIG proxy.config.reverse_proxy.enabled INT 0
CONFIG proxy.config.url_remap.remap_required INT 0
CONFIG proxy.config.url_remap.pristine_host_hdr INT 0
CONFIG proxy.config.http.response_server_enabled INT 0

// 配置443端口为安全端口
CONFIG proxy.config.http.server_ports STRING 8080 8080:ipv6 443:ssl

CONFIG proxy.config.http.connect_ports STRING 443 6650-6660

// CA根证书
CONFIG proxy.config.ssl.CA.cert.filename STRING ca.cert.pem
CONFIG proxy.config.ssl.CA.cert.path STRING /opt/proxy

// ATS服务证书目录
CONFIG proxy.config.ssl.server.cert.path STRING /opt/proxy
```

- /opt/ts/etc/trafficserver/ssl_multicert.config
```
dest_ip=* ssl_cert_name=proxy.cert.pem ssl_key_name=proxy.key.pem
```

- /opt/ts/etc/trafficserver/sni.config
此配置为路由表，Proxy根据此表会把客户端的请求转发到"tunnel_route"指定的地址
```
sni:
  - fqdn: 10000.fate.org
    tunnel_route: 192.168.0.2:6651
  - fqdn: 9999.fate.org
    tunnel_route: 192.168.0.3:6651

```
更多关于配置文件的详细描述请参考[官方文档](https://docs.trafficserver.apache.org/en/9.0.x/admin-guide/configuring-traffic-server.en.html)。

2. 启动服务
把前面步骤中为ATS生成("proxy.fate.org"目录下)的证书、私钥以及CA的证书拷贝到主机的"/opt/proxy"目录下，并使用以下命令启动ATS:
```
/opt/ts/bin/trafficserver start
```

#### 部署Pulsar
Pulsar的部署在[pulsar_deployment_guide](common/pulsar_deployment_guide.zh.md)详细描述，只需要在其基础上为broker添加证书以及打开安全服务端口，具体操作如下：
1. 登录相应主机，把为10000.fate.org生成的证书、私钥以及CA证书拷贝到"/opt/pulsar/certs"目录下

2. 修改pulsar安装目录下的conf/standalone.conf文件，增加以下内容
```
brokerServicePortTls=6651
webServicePortTls=8081
tlsEnabled=true
tlsAllowInsecureConnection=true
tlsCertificateFilePath=/opt/pulsar/certs/broker.cert.pem
tlsKeyFilePath=/opt/pulsar/certs/broker.key-pk8.pem
tlsTrustCertsFilePath=/opt/pulsar/certs/ca.cert.pem
bookkeeperTLSTrustCertsFilePath=/opt/pulsar/certs/ca.cert.pem
brokerClientTlsEnabled=true
```

3. 启动pulsar
```
$ pulsar standalone -nss
```
主机9999.fate.org上的pulsar服务也以同样步骤启动。

#### 更新FATE的路由表

- 在10000的`conf/pulsar_route_table.yaml`中更新`default`域如下:
```yml
  
10000:
  host: 192.168.0.2
  port: 6650

default:
  proxy: "proxy.fate.org:443"
  domain: "fate.org"
```

- 在9999的`conf/pulsar_route_table.yaml`中更新default域如下:
```yml
9999:
  host: 192.168.0.3
  port: 6650

default:
  proxy: "proxy.fate.org:443"
  domain: "fate.org"
```

当完成以上配置后，FATE在为目标party创建用于同步的pulsar集群时会自动根据`default`域的内容填充集群的`host`和`proxy`等参数，例如，在party 10000中用于与party 9999同步的pulsar集群信息如下:
```
{
  "serviceUrl" : "",
  "serviceUrlTls" : "",
  "brokerServiceUrl" : "pulsar://9999.fate.org:6650",
  "brokerServiceUrlTls" : "pulsar+ssl://9999.fate.org:6651",
  "proxyServiceUrl" : "pulsar+ssl://proxy.fate.org:443",
  "proxyProtocol" : "SNI",
  "peerClusterNames" : [ ]
}
```

至此，星型部署完毕，如需要增加参与方则签发为参与方签发新证书并增加路由即可。
