# Federated-Benchmark: A Benchmark of Real-world Images Dataset for Federated Learning

## Overview
We present a real-world image dataset, reflecting the characteristic real-world federated learning scenarios, and provide an extensive benchmark on model performance, efficiency, and communication in a federated learning setting.

## Resources
* Dataset: [dataset.fedai.org](https://dataset.fedai.org)
* Paper: ["Real-World Image Datasets for Federated Learning"](https://arxiv.org/abs/1910.11089)

### Street_Dataset
* Overview: Image Dataset
* Details: 7 different classes, 956 images with pixels of 704 by 576, 5 or 20 devices
* Task: Object detection for federated learning
* [Dataset_description.md](https://github.com/FederatedAI/FATE/blob/master/research/federated_object_detection_benchmark/README.md)

## Getting Started
We implemented two mainstream object detection algorithms (YOLOv3 and Faster R-CNN). Code for YOLOv3 is borrowed from  [PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3.git) and Faster R-CNN from [simple-faster-rcnn-pytorch](https://github.com/chenyuntc/simple-faster-rcnn-pytorch.git).
### Install dependencies
* requires PyTorch with GPU (code are GPU only)
* install cupy, you can install via `pip install cupy-cuda80` or (cupy-cuda90, cupy-cuda101, etc)
* Optional but strongly recommended: build cython code `nms_gpu_post`:
    ```bash
    cd cv_task/model/utils/
    python build.py build_ext --inplace
    cd -
    ```
### Prepare data
1. Download the dataset, refer to [dataset.fedai](https://dataset.fedai.org/)
2. Extract the content, rename the dir to `Street_Dataset` and put it under `FATE/cv_task/data/`
3. It should have the basic structure
    ```bash
   FATE/cv_task/data/Street_Dataset/Images
   FATE/cv_task/data/Street_Dataset/train_label.json
   FATE/cv_task/data/Street_Dataset/test_label.json
    ```
4. Generate config file for federated learning
    * Pascal VOC format for Faster R-CNN
        ```bash
        cd FATE/cv_task/data/
        python generate_task_json.py
        ```
    * YOLO format for YOLOv3
        ```bash
         python voc_label.py
        ```
5. DSL file and runtime config file for federated learning
    * DSL : `examples/federatedml-1.x-examples/homo_nn/cv.json`
    * runtime_config: `examples/federatedml-1.x-examples/homo_nn/test_homo_nn_train_then_predict.json`
    * Modify the `examples/federatedml-1.x-examples/quick_run.py`
        ```bash
        DSL_PATH = 'homo_nn/test_homo_nn_train_then_predict.json'
        SUBMIT_CONF_PATH = 'homo_nn/cv.json'
        GUEST_DATA_SET = 'breast_homo_guest.csv'
        HOST_DATA_SET = 'breast_homo_host.csv'
        # GUEST_DATA_SET and HOST_DATA_SET are needed by quick_run.py
        # The image dataset will be read from local path
        ```    
### Train
1. Start server
2. Run the job
    ```bash
    python quick_run.py
    ```
3. If you want to run on GPUs, you should modify `fate_flow/service.sh` to use your own `python` to start server.
### Citation
* If you use this code or dataset for your research, please kindly cite our paper:
```bash
@article{luo2019real,
  title={Real-World Image Datasets for Federated Learning},
  author={Luo, Jiahuan and Wu, Xueyang and Luo, Yun and Huang, Anbu and Huang, Yunfeng and Liu, Yang and Yang, Qiang},
  journal={arXiv preprint arXiv:1910.11089},
  year={2019}
}
```