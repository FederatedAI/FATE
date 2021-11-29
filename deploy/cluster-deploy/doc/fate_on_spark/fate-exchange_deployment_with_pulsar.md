## FATE Exchange with Pulsar Deployment Guide
[中文](fate-exchange_deployment_with_pulsar.zh.md)

## Star Networking

Using pulsar as a transport service can support star deployment, its central node is a SNI (Server Name Indication) proxy service, the specific proxy service can use [Apache Traffic Server](https://trafficserver.apache.org/). The specific proxy process is as follows.
1. The client sends a TLS Client Hello request to the proxy server with a SNI field that declares the domain name or host name of the remote server to which the client wants to connect.
2. The proxy server establishes a TCP tunnel with the remote server based on the SNI field and its own routing information and forwards the client's TLS Hello. 3.
3. The remote server sends the TLS Server Hello to the client and then completes the TLS handshake. 4.
4. TCP link is established and the client and remote server communicate normally.

### Specific deployment method
The next step is to build a federated learning network based on the SNI proxy model. Since it involves the generation of certificates, the network can be identified by a unified domain name suffix, such as "fate.org". The entities in the network can then be identified by `${party_id}.fate.org`, e.g. party 10000 uses a certificate with CN "10000.fate.org".

#### Planning
Hostname | IP Address | Operating System | Installed Software | Services
-------|--------|----------|----------|-----
proxy.fate.org | 192.168.0.1 | CentOS 7.2/Ubuntu 16.04 | ats | ats
10000.fate.org | 192.168.0.2 | CentOS 7.2/Ubuntu 16.04 | pulsar | pulsar
9999.fate.org | 192.168.0.3 | CentOS 7.2/Ubuntu 16.04 | pulsar | pulsar

The specific architecture is shown below. The pulsar service "10000.fate.org" belongs to the organization with ID 10000, while the pulsar service "9999.fate.org" belongs to the organization with ID 9999, and the "proxy.fate.org" is the ats service, which is the center of the star network.
<div style="text-align:center", align=center>
<img src="../../images/pulsar_sni_proxy.png" />
</div>

#### Certificate Generation
Since the SNI proxy is based on TLS, you need to configure certificates for the ATS and pulsar services. The first thing you need to do is to generate CA certificates and then issue certificates for the ats and pulsar services with the same "CN" as their hostname (in reality the "CN" of the certificate can be different from the hostname).

##### Generate a CA certificate
Enter the following command to create a directory for the CA and place this openssl configuration file in that directory.
``` bash
$ mkdir my-ca
$ cd my-ca
$ wget https://raw.githubusercontent.com/apache/pulsar/master/site2/website/static/examples/openssl.cnf
$ export CA_HOME=$(pwd)
```

Enter the following command to create the necessary directories, keys and certificates.
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
In the above command, further interaction is required to generate the key and certificate, so the user can enter them according to the prompts. Once the above command has been run, the CA related certificate and key are generated.
Among them.
- certs/ca.cert.pem holds the CA certificate file
- private/ca.key.pem saves the CA key file

##### Generate certificate for 10000.fate.org
1. generate the directory to store the certificate file
```
$ mkdir 10000.fate.org
``` 

2. Enter the following command to generate the key.
```
$ openssl genrsa -out 10000.fate.org/broker.key.pem 2048
``` 

3. The Broker needs the key to be in PKCS 8 format, so enter the following command to convert it.
```
$ openssl pkcs8 -topk8 -inform PEM -outform PEM \
      -in 10000.fate.org/broker.key.pem -out 10000.fate.org/broker.key-pk8.pem -nocrypt
```

4. Enter the following command to generate a certificate request, where `Common Name` is entered as **10000.fate.org**
```
$ openssl req -config openssl.cnf \
    -key 10000.fate.org/broker.key.pem -new -sha256 -out 10000.fate.org/broker.csr.pem
```

5. Enter the following command to obtain the signature of the certificate authority.
```
$ openssl ca -config openssl.cnf -extensions server_cert \
    -days 1000 -notext -md sha256 \
    -in 10000.fate.org/broker.csr.pem -out 10000.fate.org/broker.cert.pem
```
At this time, the "10000.fate.org" directory stores the certificate "broker.cert.pem" and a key "broker.key-pk8.pem". At this point the client can work with the CA certificate to verify the broker service.

##### generates a certificate for 9999.fate.org
The "9999.fate.org" certificate is generated in the same way as the above steps, and the `Common Name` in step 4 is entered as **9999.fate.org**.

The following operation will default the certificate of "9999.fate.org" has been generated and placed in the directory of "9999.fate.org".


##### Generate certificate for proxy.fate.org
The certificate for "proxy.fate.org" is generated in the same way as the above steps, the conversion in part 3 can be omitted, and the `Common Name` in step 5 is entered as **proxy.fate.org**.

The following operation will default the certificate of "proxy.fate.org" has been generated and placed in the directory of "proxy.fate.org", the certificate and private key are "proxy.cert.pem" and "proxy.key.pem" respectively

#### Deploying Apache Traffic Server
##### Installing Apache Traffic Server

1. Log in to the "proxy.fate.org" host and prepare the dependencies according to this [documentation](https://github.com/apache/trafficserver/tree/9.0.0) depending on the operating system.

2. Download Apache Traffic server 9.0
```
$ wget https://apache.claz.org/trafficserver/trafficserver-9.0.0.tar.bz2
```

3. Unzip and install
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

When the command is executed, the traffic server will be installed in the `/opt/ts` directory and the path of the profile will be `/opt/ts/etc/trafficserver/`.

##### Start Apache Traffic Server service
1. Modify the ATS configuration
- /opt/ts/etc/trafficserver/records.config
```
CONFIG proxy.config.http.cache.http INT 0
CONFIG proxy.config.reverse_proxy.enabled INT 0
CONFIG proxy.config.url_remap.remap_required INT 0
CONFIG proxy.config.url_remap.pristine_host_hdr INT 0
CONFIG proxy.config.http.response_server_enabled INT 0

// Configure port 443 as a secure port
CONFIG proxy.config.http.server_ports STRING 8080 8080:ipv6 443:ssl

CONFIG proxy.config.http.connect_ports STRING 443 6650-6660

// CA root certificate
CONFIG proxy.config.ssl.CA.cert.filename STRING ca.cert.pem
CONFIG proxy.config.ssl.CA.cert.path STRING /opt/proxy

// ATS service certificate directory
CONFIG proxy.config.ssl.server.cert.path STRING /opt/proxy
```

- /opt/ts/etc/trafficserver/ssl_multicert.config
```
dest_ip=* ssl_cert_name=proxy.cert.pem ssl_key_name=proxy.key.pem
```

- /opt/ts/etc/trafficserver/sni.config
This configuration is the routing table, according to which the Proxy will forward the client requests to the address specified by "tunnel_route"
```
sni:
  - fqdn: 10000.fate.org
    tunnel_route: 192.168.0.2:6651
  - fqdn: 9999.fate.org
    tunnel_route: 192.168.0.3:6651

```
For more detailed description of the configuration file, please refer to the [official documentation](https://docs.trafficserver.apache.org/en/9.0.x/admin-guide/configuring-traffic-server.en.html).

2. Start the service
Copy the certificate, private key and CA's certificate generated for the ATS in the previous step (in the "proxy.fate.org" directory) to the "/opt/proxy" directory of the host, and start the ATS with the following command:
```
/opt/ts/bin/trafficserver start
```

#### Deploying Pulsar
Pulsar is deployed in [pulsar_deployment_guide](common/pulsar_deployment_guide.md) is described in detail and only requires adding a certificate for the broker and opening the secure service port on top of it, as follows.
1. Log in to the corresponding host and copy the certificate, private key and CA certificate generated for 10000.fate.org to the "/opt/pulsar/certs" directory

2. Modify the conf/standalone.conf file in the pulsar installation directory and add the following contents
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

3. Start pulsar
```
$ pulsar standalone -nss
```
Start the pulsar service on host 9999.fate.org with the same steps.

#### Update the routing table of FATE

- Update the ``default`` field in `conf/pulsar_route_table.yaml` in 10000 as follows:
```yml
  
10000:
  host: 192.168.0.2
  port: 6650

default:
  proxy: "proxy.fate.org:443"
  domain: "fate.org"
```

- Update the default domain in `conf/pulsar_route_table.yaml` in 9999 as follows:
```yml
9999:
  host: 192.168.0.3
  port: 6650

default:
  proxy: "proxy.fate.org:443"
  domain: "fate.org"
```

When the above configuration is done, FATE will automatically populate the `host` and `proxy` parameters of the cluster based on the `default` domain when creating the pulsar cluster for the target party to synchronize with, e.g., the pulsar cluster used to synchronize with party 9999 in party 10000 will have the following information:
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

At this point, the star deployment is complete, if you need to add participants then issue a new certificate for the participant and add routes.
