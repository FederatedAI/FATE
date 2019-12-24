#进入examples/test文件夹<br>
===========================
执行命令python run_test.py env.json result.txt<br>
---------------------------------------
run_test.py测试文件，搜寻并执行测试任务<br>
env.json环境文件，指明所需要的环境配置，与conf文件类似<br>  
result.txt输出文件，指明查看结果的位置，用户自定义<br>

其中执行的任务为examples/federatedml-1.x-examples文件夹中以testsuite.json结尾设置的任务<br>
一个testsuite.json样例已经给出，在temp_testsuite.json包括一个训练和一个预测任务.<br>
事实上，只要任务文件在examples/federatedml-1.x-examples/中以testsuite.json结尾都会执行<br>

env.json文件说明<br>
"role"指明角色id，包括host，guest，以及arbiter<br>
"ip_map"构建角色与实际ip的映射，本地为-1，远程的主机为实际的ip地址<br>

testsuite.json文件说明<br>

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
支持多个任务，列表中可以一次性上传多个数据。<br>
role字段注明角色，以及在环境配置文件中的对应关系，guest_0代表数据在环境文件中配置的guest列表中的第一个主机上。<br>
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
  支持训练任务和预测任务，格式略有区别。<br>
  预测任务需要在task字段指明产生模型的训练任务。<br>
  请用不同的名字命名不同的任务，重复名字的任务，只会得到最后提交的任务结果。<br>
  一个成功的任务结果示例
  
  一个失败的任务结果示例
