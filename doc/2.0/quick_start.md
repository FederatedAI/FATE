## Quick Start

1. install `fate_client` with extra package `fate`  

```sh
python -m pip install -U pip && python -m pip install fate_client[fate,fate_flow]==2.0.0b0
```
after installing packages successfully, initialize fate_flow service and fate_client

```sh
mkdir fate_workspace
fate_flow init --ip 127.0.0.1 --port 9380 --home $(pwd)/fate_workspace
pipeline init --ip 127.0.0.1 --port 9380
```


2.  download example data

```sh
wget https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_guest.csv && \
wget https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_host.csv
```

3. transform example data to dataframe using in fate
```python
import os
from fate_client.pipeline import FateFlowPipeline


base_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
guest_data_path = os.path.join(base_path, "breast_hetero_guest.csv")
host_data_path = os.path.join(base_path, "breast_hetero_host.csv")

data_pipeline = FateFlowPipeline().set_roles(local="0")
guest_meta = {
    "delimiter": ",", "dtype": "float64", "label_type": "int64","label_name": "y", "match_id_name": "id"
}
host_meta = {
    "delimiter": ",", "input_format": "dense", "match_id_name": "id"
}
data_pipeline.transform_local_file_to_dataframe(file=guest_data_path, namespace="experiment", name="breast_hetero_guest",
                                                meta=guest_meta, head=True, extend_sid=True)
data_pipeline.transform_local_file_to_dataframe(file=host_data_path, namespace="experiment", name="breast_hetero_host",
                                                meta=host_meta, head=True, extend_sid=True)
```
4. run example 

```python
from fate_client.pipeline.components.fate import (
    HeteroSecureBoost,
    PSI,
    Evaluation
)
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.interface import DataWarehouseChannel


# create pipeline for training
pipeline = FateFlowPipeline().set_roles(guest="9999", host="10000")

# create psi component_desc
psi_0 = PSI("psi_0")
psi_0.guest.component_setting(
    input_data=DataWarehouseChannel(name="breast_hetero_guest", namespace="experiment")
)
psi_0.hosts[0].component_setting(
    input_data=DataWarehouseChannel(name="breast_hetero_host", namespace="experiment")
)

# create hetero secure_boost component_desc
hetero_secureboost_0 = HeteroSecureBoost(
    'hetero_secureboost_0', num_trees=1, max_depth=5,
    train_data=psi_0.outputs['output_data'],
    validate_data=psi_0.outputs["output_data"]
)

# create evaluation component_desc
evaluation_0 = Evaluation(
    'evaluation_0', runtime_roles=['guest'], metrics=['auc'], input_data=[hetero_secureboost_0.outputs['train_data_output']]
)

# add training task
pipeline.add_task(psi_0)
pipeline.add_task(hetero_secureboost_0)
pipeline.add_task(evaluation_0)

# compile and train
pipeline.compile()
pipeline.fit()

# print metric and model info
print (pipeline.get_task_info("hetero_secureboost_0").get_output_model())
print (pipeline.get_task_info("evaluation_0").get_output_metric())

# deploy task for inference
pipeline.deploy([psi_0, hetero_secureboost_0])

# create pipeline for predicting
predict_pipeline = FateFlowPipeline()

# add input to deployed_pipeline
deployed_pipeline = pipeline.get_deployed_pipeline()
deployed_pipeline.psi_0.guest.component_setting(
    input_data=DataWarehouseChannel(name="breast_hetero_guest", namespace=f"experiment")
)
deployed_pipeline.psi_0.hosts[0].component_setting(
    input_data=DataWarehouseChannel(name="breast_hetero_host", namespace=f"experiment")
)

# add task to predict pipeline
predict_pipeline.add_task(deployed_pipeline)

# compile and predict
predict_pipeline.compile()
predict_pipeline.predict()
```
