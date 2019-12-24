进入examples/test文件夹
执行命令python run_test.py env.json result.txt
run_test.py测试文件，搜寻并执行测试任务
env.json环境文件，指明所需要的环境配置，与conf文件类似  
result.txt输出文件，指明查看结果的位置，用户自定义

其中执行的任务为examples/federatedml-1.x-examples文件夹中以testsuite.json结尾设置的任务
一个testsuite.json样例已经给出，在temp_testsuite.json包括一个训练和一个预测任务.
事实上，只要任务文件在examples/federatedml-1.x-examples/中以testsuite.json结尾都会执行

env.json文件说明
"role"指明角色id，包括host，guest，以及arbiter
"ip_map"构建角色与实际ip的映射，本地为-1，远程的主机为实际的ip地址

testsuite.json文件说明

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
支持多个任务，列表中可以一次性上传多个数据。
role字段注明角色，以及在环境配置文件中的对应关系，guest_0代表数据在环境文件中配置的guest列表中的第一个主机上。
  "tasks": {

    "lr": {

      "conf": "../predicttask/conf.json",

      "dsl": "../predicttask/dsl.json",

      "type": "train"

    },

    "lr-predict": {

      "conf": "../predicttask/predict.json",

      "task": "lr",

      "type": "predict"

    }

  }
  支持训练任务和预测任务，格式略有区别。
  预测任务需要在task字段指明产生模型的训练任务。
  请用不同的名字命名不同的任务，重复名字的任务，只会得到最后提交的任务结果。
