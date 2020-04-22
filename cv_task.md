cv_task使用说明
===============
文件目录结构
------------
  cv_task中保留了原有工程绝大部分文件，起到库的作用，包括数据处理dataloader_detector.py，构建网络net.py等。<br>
  fate_flow为程序执行的目录，为了方便，将数据文件夹luna_npy放到该目录下。<br>
  新的执行逻辑主要在federatedml/nn/homo_nn/enter_point.py中，主要用于执行加载数据，调用模型训练，聚合，以及预测任务。<br>

可选择配置的文件
-----------------
  examples/federatedml-1.x-examples/homo_nn/cv.json中，之前通过argparse生成config_default，在本文件完成。<br>
 ``` 
 "nn_define": [
       {
        "workers": 1,
        "epochs":1,
        "batch_size":1,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay":1e-4,
        "split": 1,
        "gpu":"none",
        "validation_subset": 0,
        "training_subset": 0
      }
      ]
```
  其余位置无需修改。<br>

执行任务
----------
```
python fate_flow_client.py -f submit_job -d../examples/federatedml-1.x-examples/homo_nn/test_homo_nn_train_then_predict.json  -c ../examples/federatedml-1.x-examples/homo_nn/cv.json 
```
