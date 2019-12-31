
测试工具使用说明
================
1.使用方法
------
进入examples/test文件夹。<br>

执行命令python run_test.py default_env.json <br>

run_test.py执行测试文件，搜寻并执行测试任务。<br>
env.json环境文件，根据用户实际情况指明所需要的环境配置。<br>  

可选参数

 "-o" 指定输出结果文件，默认为 `test_result`",

 "-e",指定错误记录文件
    
 "-i", "--interval" 指定查询任务状态间隔
 
 "--skip_data", 跳过数据上传，默认为false

互斥可选参数
 
  "-d", "--dir"  指定多个任务所在的根目录

  "-s", "--suite" 指定执行某个任务文件

2.执行规则
---------
在没有指定name参数的情况下，测试工具执行的任务为examples/federatedml-1.x-examples文件夹中以testsuite.json为后缀的任务文件。<br>
指定后，测试工具执行的任务为examples/federatedml-1.x-examples文件夹中以name参数为后缀的任务文件。<br>
一个testsuite.json任务文件样例已经给出。<br>
examples/test/temp_testsuite.json<br>
在temp_testsuite.json中包括一个训练和一个预测任务。<br>

3.文件说明
-----------
env.json<br>
"role"指明角色id，包括host，guest，以及arbiter.<br>
"ip_map"构建角色与实际ip的映射，本地为-1，远程的主机为实际的ip地址。<br>

testsuite.json<br>
data字段支持多个任务，在列表中可以一次性上传多个以字典形式配置的数据。<br>
其中role字段建立与env.json的联系，guest_0代表数据在env.json中配置的guest列表中的第一个主机上。<br>
tasks配置需要执行的任务，目前支持训练任务和预测任务，格式略有区别。<br>
预测任务需要在task字段指明产生模型的训练任务名。<br>
请用不同的名字命名不同的任务，重复名字的任务，只会得到最后配置的任务结果。<br>

例子：
```shell script
python run_test.py default_env.json -s ./demo/temp_testsuite.json
```
4.结果示例
-----------
一个成功的任务结果示例<br>
lr（任务名）     201912241035408383043（job_id）success（状态）<br>
一个失败的任务结果示例<br>
lr-predict      201912241039146131304failed

Instructions of test tools
=================
1.Use 
------
Execute commands<br>
cd examples/test<br>
python run_test.py default_env.json<br>

run_test.py  search and execute tasks defined by users. <br>
env.json environment configs based on users' running environment. <br>

Optional parameters
 "-o", "--output", "file to save result, defaults to `test_result`" <br>
 "-e", "--error", "file to save error" <br>
 "-i", "--interval", "check job status every i seconds, defaults to 3" <br>
 "--skip_data", "skip data upload, used to be false if not use <br>
 
 mutually_exclusive_group include: <br>
 "-d", "--dir", "dir to find testsuites", <br>
 "-s", "--suite","a single testsuite to run" <br>


2.Tips
------
If name is not given,the script will execute the tasks defined in task files from examples/federatedml-1.x-examples folder with a "testsuite.json" suffix.<br>
If there is a name parameter,the script will execute the tasks defined in task files from examples/federatedml-1.x-examples folder with a suffix given by name.
An example task file is given in examples/test/temp_testsuite.json including a training and a prediction task. <br>

3.Config files
------
env.json <br>
Please set role id in "role", including host, guest, and arbiter.<br>
Please build the relationship between roles and ip in "ip_map",where -1 stands for local,and remote host will be given ip address. <br>

testsuite.json <br>
You can submit data for many tasks once in "data",and each has a series of configs in a dict.<br>
"role" parameter describes the location of the data defined in env.json.For example, "guest_0" represents the data located in the first guest defined in the guest list of env.json. <br>
You can define your own tasks in "tasks".Training tasks and prediction tasks are supported now. There is some difference between them.<br>
A prediction task needs to state the task name of the training task which it depends on. <br>
Please name different tasks with different names,if two tasks share the same name,you will get the execution result of the letter defined. <br>

demo:<br>
```shell script
python run_test.py default_env.json -s ./demo/temp_testsuite.json
```
4.Examples of results
------

```text
./demo/temp_testsuite.json
====================================================================
lr	success	201912271619411350983
lr-predict	success	201912271620429623264
```


