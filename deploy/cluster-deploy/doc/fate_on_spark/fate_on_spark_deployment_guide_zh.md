# FATE ON Spark 部署指南

## 1.服务器配置

|  服务器  |                                                              |
| :------: | ------------------------------------------------------------ |
|   数量   | >1（根据实际情况配置）                                       |
|   配置   | 8 core /16GB memory / 500GB硬盘/10M带宽                      |
| 操作系统 | CentOS linux 7.2及以上/Ubuntu 16.04 以上                     |
|  依赖包  | （参见4.5 软件环境初始化）                                   |
|   用户   | 用户：app，属主：apps（app用户需可以sudo su root而无需密码） |
| 文件系统 | 1.  500G硬盘挂载在/ data目录下； 2.创建/ data / projects目录，目录属主为：app:apps |

## 2.集群规划

| party  | partyid | 主机名        | IP地址      | 操作系统                | 安装软件    | 服务                              |
| ------ | ------- | ------------- | ----------- | ----------------------- | ----------- | --------------------------------- |
| PartyA | 10000   | VM-0-1-centos | 192.168.0.1 | CentOS 7.2/Ubuntu 16.04 | fate，mysql, nginx | fateflow，fateboard，mysql，nginx |
| PartyA | 10000   |               |             |                         | Spark、HDFS |                                   |
| PartyA | 10000   |               |             |                         | RabbitMQ    |                                   |
| PartyB | 9999    | VM-0-2-centos | 192.168.0.2 | CentOS 7.2/Ubuntu 16.04 | fate，mysql, nginx | fateflow，fateboard，mysql，nginx |
| PartyB | 9999    |               |             |                         | Spark、HDFS |                                   |
| PartyB | 9999    |               |             |                         | RabbitMQ    |                                   |

架构图：

<div style="text-align:center", align=center>
<img src="../../images/fate_on_spark_architecture.png" />
</div>

## 3.组件说明

| 软件产品 | 组件      | 端口      | 说明                                                  |
| -------- | --------- | --------- | ----------------------------------------------------- |
| fate     | fate_flow | 9360;9380 | 联合学习任务流水线管理模块，每个party只能有一个此服务 |
| fate     | fateboard | 8080      | 联合学习过程可视化模块，每个party只能有一个此服务     |
| nginx    | nginx     | 9370      | 跨站点(party)调度协调代理                             |
| mysql    | mysql     | 3306      | 元数据存储                                            |
| Spark    |           |           | 计算引擎                                              |
| HDFS     |           |           | 存储引擎                                              |
| RabbitMQ |           |           | 跨站点(party)数据交换代理                             |

## 4 部署Spark & HDFS
请参阅部署指南：[Hadoop_Spark_deployment_guide_zh](hadoop_spark_deployment_guide_zh.md)

## 5. 部署FATE
请参阅部署指南：[fate_deployment_step_by_step_zh](fate_deployment_step_by_step_zh.md)的1、2、3章节

## 6. FATE配置文件修改
请参阅部署指南：[fate_deployment_step_by_step_zh](fate_deployment_step_by_step_zh.md)的第4章节
其中hdfs的namenode配置为对应的

## 7. 启动
请参阅部署指南：[fate_deployment_step_by_step_zh](fate_deployment_step_by_step_zh.md)的第5章节

## 8. 问题定位
请参阅部署指南：[fate_deployment_step_by_step_zh](fate_deployment_step_by_step_zh.md)的第6章节

## 9. 测试
请参阅部署指南：[fate_deployment_step_by_step_zh](fate_deployment_step_by_step_zh.md)的第7章节

## 10.系统运维
请参阅部署指南：[fate_deployment_step_by_step_zh](fate_deployment_step_by_step_zh.md)的第8章节

## 11. 附录
请参阅部署指南：[fate_deployment_step_by_step_zh](fate_deployment_step_by_step_zh.md)的第9章节
