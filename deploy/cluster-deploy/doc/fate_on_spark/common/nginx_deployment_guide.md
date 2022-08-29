## Nginx Deployment Guide

### 1. dependency download
```shell script
mkdir -r /data/projects/install && cd /data/projects/install
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/resources/openresty-1.17.8.2.tar.gz
```

### 2. Deployment

```bash
cd /data/projects/install
tar xzf openresty-*.tar.gz
cd openresty-*
./configure --prefix=/data/projects/fate/proxy \
                   --with-luajit \
                   --with-http_ssl_module \
                     --with-http_v2_module \
                     --with-stream \
                     --with-stream_ssl_module \
                     -j12
make && make install
```

### 3. Configuration modifications
#### 3.1 Nginx base configuration file modifications
Configuration file: /data/projects/fate/proxy/nginx/conf/nginx.conf
This configuration file is used by Nginx to configure the service base settings and lua code, and generally does not need to be modified.
If you want to modify it, you can manually modify it by referring to the default nginx.conf, and use the command detect after the modification is done
```
/data/projects/fate/proxy/nginx/sbin/nginx -t
```

#### 3.2 Nginx routing configuration file modification (after deploying fate)

Configuration file: /data/projects/fate/proxy/nginx/conf/route_table.yaml
This configuration file is used by Nginx to configure routing information, either manually by referring to the following example, or by using the following command.

```
# Modify the execution under the target server (192.168.0.1) app user
cat > /data/projects/fate/proxy/nginx/conf/route_table.yaml << EOF
default:
  proxy:
    - host: 192.168.0.2
      http_port: 9300
      grpc_port: 9310
10000:
  proxy:
    - host: 192.168.0.1
      http_port: 9300
      grpc_port: 9310
  fateflow:
    - host: 192.168.0.1
      http_port: 9380
      grpc_port: 9360
9999:
  proxy:
    - host: 192.168.0.2
      http_port: 9300
      grpc_port: 9310
  fateflow:
    - host: 192.168.0.2
      http_port: 9380
      grpc_port: 9360
EOF

# Modify the execution under the app user of the target server (192.168.0.2)
cat > /data/projects/fate/proxy/nginx/conf/route_table.yaml << EOF
default:
  proxy:
    - host: 192.168.0.1
      http_port: 9300
      grpc_port: 9310
10000:
  proxy:
    - host: 192.168.0.1
      http_port: 9300
      grpc_port: 9310
  fateflow:
    - host: 192.168.0.1
      http_port: 9380
      grpc_port: 9360
9999:
  proxy:
    - host: 192.168.0.2
      http_port: 9300
      grpc_port: 9310
  fateflow:
    - host: 192.168.0.2
      http_port: 9380
      grpc_port: 9360
EOF
```

### 4. startup and logging module
#### 4.1 Starting the service
```
cd /data/projects/fate/proxy
./nginx/sbin/nginx -c /data/projects/fate/proxy/nginx/conf/nginx.conf
```

#### 4.2 Log directory
```
/data/projects/fate/proxy/nginx/logs
```
