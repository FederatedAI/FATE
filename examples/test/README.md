
测试工具使用说明
================
1.使用方法
------
进入examples/test文件夹<br>
执行命令python run_test.py env.json result.txt<br>

run_test.py执行测试文件，搜寻并执行测试任务<br>
env.json环境文件，根据用户实际情况指明所需要的环境配置<br>  
result.txt输出文件，指明查看结果的位置<br>

2.执行规则
---------
测试工具执行的任务为examples/federatedml-1.x-examples文件夹中以testsuite.json为后缀的任务文件<br>
一个testsuite.json任务文件样例已经给出<br>
examples/test/temp_testsuite.json<br>
在temp_testsuite.json中包括一个训练和一个预测任务.<br>

3.文件说明
-----------
env.json<br>
{
    "role": {
      "guest": [],
      "host": [],
      "arbiter": []
    },
    "ip_map": {}  
}
"role"指明角色id，包括host，guest，以及arbiter<br>
"ip_map"构建角色与实际ip的映射，本地为-1，远程的主机为实际的ip地址<br>

testsuite.json<br>
  "data": [

       {
          "file": "examples/data/breast_b.csv",

          "head": 1,

          "partition": 10,

          "table_name": "hetero_breast_b",

          "namespace": "hetero_breast_guest",

          "role": "guest_0"

        }
 ]
data字段支持多个任务，在列表中可以一次性上传多个以字典形式配置的数据。<br>
其中role字段建立与env.json的联系，guest_0代表数据在env.json中配置的guest列表中的第一个主机上。<br>
  "tasks": {

    "lr": {

      "conf": "train_job_conf.json",

      "dsl": "train_job_dsl.json",

      "type": "train"

    },

    "lr-predict": {

      "conf": "predict_conf.json",

      "task": "lr",

      "type": "predict"

    }

  }
  tasks配置需要执行的任务，目前支持训练任务和预测任务，格式略有区别。<br>
  预测任务需要在task字段指明产生模型的训练任务名。<br>
  请用不同的名字命名不同的任务，重复名字的任务，只会得到最后配置的任务结果。<br>
 4.结果示例
 -----------
  一个成功的任务结果示例<br>
  lr（任务名）              201912241035408383043（job_id）success（状态）<br>
  一个失败的任务结果示例<br>
  lr-predict      201912241039146131304failed
