## Nginx部署指南

### 1. 依赖下载
```shell script
mkdir -r /data/projects/install && cd /data/projects/install
wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/resources/openresty-1.17.8.2.tar.gz
```

### 2. 部署

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

### 3. 配置修改
#### 3.1 Nginx基础配置文件修改
配置文件:  /data/projects/fate/proxy/nginx/conf/nginx.conf
此配置文件Nginx使用，配置服务基础设置以及lua代码，一般不需要修改。
若要修改，可以参考默认nginx.conf手工修改，修改完成后使用命令检测
```
/data/projects/fate/proxy/nginx/sbin/nginx -t
```

#### 3.2 Nginx路由配置文件修改(需要部署完fate)

配置文件:  /data/projects/fate/proxy/nginx/conf/route_table.yaml
此配置文件Nginx使用，配置路由信息，可以参考如下例子手工配置，也可以使用以下指令完成：

```
#在目标服务器（192.168.0.1）app用户下修改执行
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

#在目标服务器（192.168.0.2）app用户下修改执行
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

### 4. 启动及日志模块
#### 4.1 启动服务
```
cd /data/projects/fate/proxy
./nginx/sbin/nginx -c /data/projects/fate/proxy/nginx/conf/nginx.conf
```

#### 4.2 日志目录
```
/data/projects/fate/proxy/nginx/logs
```
