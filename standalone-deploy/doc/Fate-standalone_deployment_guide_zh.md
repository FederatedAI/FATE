## ****FATE单机部署指南****

1.服务器配置
============

|  服务器  |                                                              |
| :------: | ------------------------------------------------------------ |
|   数量   | 1                                                            |
|   配置   | 8 core /16GB memory / 500GB硬盘/10M带宽                      |
| 操作系统 | CentOS linux 7.2及以上 / Ubuntu 16.04以上                    |
|   用户   | 用户：app，属主：apps（app用户需可以sudo su root而无需密码） |
| 文件系统 | 1.  500G硬盘挂载在/ data目录下； 2.创建/ data / projects目录，目录属主为：app:apps |
|   网络   | 需要良好的网络连接                                           |

#  2. 在主机中安装FATE

1. 检查本地8080、9360、9380端口是否被占用。

   ```
   netstat -apln|grep 8080
   netstat -apln|grep 9360
   netstat -apln|grep 9380
   ```

2. 下载独立版本的压缩包并解压缩。

   ```
   cd /data/projects
   wget https://webank-ai-1251170195.cos.ap-guangzhou.myqcloud.com/standalone-fate-master-1.4.0.tar.gz
   tar -xf  standalone-fate-master-1.4.0.tar.gz
   ```

3. 进入FATE目录并执行init.sh.

   ```
   cd standalone-fate-master-1.4.0
   source init.sh init
   ```

4. 执行测试.

   ```
   cd standalone-fate-master-1.4.0
   bash ./federatedml/test/run_test.sh
   ```

有些用例算法在 [examples](../../examples/federatedml-1.x-examples) 文件夹下, 请尝试使用。

您还可以通过浏览器体验算法过程看板，访问：Http://hostip:8080 。hostip代表是本机ip地址。

