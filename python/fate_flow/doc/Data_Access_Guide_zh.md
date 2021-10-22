## FATE数据接入指南

### 1. upload

#### fate提供upload组件供用户上传数据至fate所支持的存储系统内(默认eggroll)
用法如下:
```shell script
# cli
python fate_flow_client.py -f upload -c $config_path

# fate client
flow data upload -c $config_path
```
- **eggroll**

```
      python fate_flow_client.py -f upload -c examples/upload/upload_to_eggroll.json 
```
```
      flow data upload -c examples/upload/upload_to_eggroll.json 
```
- **mysql**
```
      python fate_flow_client.py -f upload -c examples/upload/upload_to_mysql.json 
```
```
      flow data upload -c examples/upload/upload_to_mysql.json 
```
- **hdfs**
```
      python fate_flow_client.py -f upload -c examples/upload/upload_to_hdfs.json 
```
```
      flow data upload -c examples/upload/upload_to_hdfs.json 
```

### 2. table bind
#### 若用户的数据已经存在fate所支持的存储系统内,可通过table bind将真实存储路径映射到fate存储表
用法如下:
```shell script
# cli
python fate_flow_client.py -f table_bind -c $config_path

# fate client
flow table bind -c $config_path
```

- **mysql**
```
      python fate_flow_client.py -f table_bind -c examples/table_bind/bind_mysql_table.json 
```
```
      flow table bind -c examples/table_bind/bind_mysql_table.json 
```
- **hdfs**
```
      python fate_flow_client.py -f table_bind -c examples/table_bind/bind_hdfs_table.json 
```
```
      flow table bind examples/table_bind/bind_hdfs_table.json
```
