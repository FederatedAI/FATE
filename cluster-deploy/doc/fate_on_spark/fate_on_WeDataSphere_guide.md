### 一. FATE部署

请参阅部署指南：[Fate_guest_install_guide_ansible](../Fate_guest_install_guide_ansible.md)


### 二. RabbitMQ部署

请参阅部署指南：[rabbitmq_deployment_guide](rabbitmq_deployment_guide_zh.md)

### 三. WeDataSphere相关组件部署

请参阅部署指南：[WeDataSphere](https://github.com/WeBankFinTech/WeDataSphere)


### 四. 修改配置

- ### linkis相关配置

  fate/conf/service_conf.yaml修改下面host、port、token_code、python_path. 
  ```yaml
  fate_on_spark:
    linkis_spark:
      cores_per_node: 20
      nodes: 2
      host: xxx.xxx.xxx.xxx
      port: 9001
      token_code: xxx
      python_path: /data/projects/fate/python
    linkis_hive:
      host: xxx.xxx.xxx.xxx
      port: 9001
  ```
  其中, host和port:linkis请求入口地址; token_code:linkis分配的Token-Code; python_path:fate依赖路径

- ### rabbitmq相关配置

  (1)fate/conf/service_conf.yaml修改下面内容为rabbitmq实际部署时的配置

  ```yaml
  fate_on_spark:
    rabbitmq:
      host: xxx.xxx.xxx.xxx
      mng_port: 12345
      port: 5672
      user: fate
      password: fate
      route_table:
  ```

  (2)fate/conf下添加新的配置文件: rabbitmq_route_table.yaml

  ```yaml
  $party_id:
    host: xxx.xxx.xxx.xxx
    port: 5672
  
  ```
  其中$party_id为当前站点fate的站点，host和port同上(1)一致

- #### 重启fate flow服务

### 五、linkis访问权限相关

- #### fate_flow数据库的操作权限需要开放给linkis机器

### 六、部署验证(fate方和WeDataSphere同时部署完成才进行验证)

- **toy**

  ```shell
  cd /data/projects/fate/examples/toy_example && source /data/projects/fate/bin/init_env.sh 
  python run_toy_example.py $party_id $party_id 1 -b 3 -v 2 -u $user_name
  ```

  其中$party_id为当前部署站点， user_name为WeDataSphere用户名

  

- **如果出现"ValueError: job running time exceed, please check federation or eggroll log", 可通过fate board或者job_query命令查询任务状态:**

  ```shell
  cd /data/projects/fate/python/fate_flow/ && python fate_flow_client.py -f query_job -j $job_id
  ```

  其中job_id为toy打印的jobid

