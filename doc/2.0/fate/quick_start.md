## Quick Start

1. install `fate_client` with extra package `fate`  

```sh
python -m pip install -U pip && python -m pip install fate_client[fate,fate_flow]==2.1.1
```
after installing packages successfully, initialize fate_flow service and fate_client

```sh
mkdir fate_workspace
fate_flow init --ip 127.0.0.1 --port 9380 --home $(pwd)/fate_workspace
pipeline init --ip 127.0.0.1 --port 9380

fate_flow start
fate_flow status # make sure fate_flow service is started
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

data_pipeline = FateFlowPipeline().set_parties(local="0")
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
4. run training example and save pipeline

```python
from fate_client.pipeline.components.fate import (
    HeteroSecureBoost,
    Reader,
    PSI,
    Evaluation
)
from fate_client.pipeline import FateFlowPipeline


# create pipeline for training
pipeline = FateFlowPipeline().set_parties(guest="9999", host="10000")

# create reader task_desc
reader_0 = Reader("reader_0")
reader_0.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
reader_0.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")

# create psi component_desc
psi_0 = PSI("psi_0", input_data=reader_0.outputs["output_data"])

# create hetero secure_boost component_desc
hetero_secureboost_0 = HeteroSecureBoost(
    "hetero_secureboost_0", num_trees=1, max_depth=5,
    train_data=psi_0.outputs["output_data"],
    validate_data=psi_0.outputs["output_data"]
)

# create evaluation component_desc
evaluation_0 = Evaluation(
    'evaluation_0', runtime_parties=dict(guest="9999"), metrics=["auc"], input_datas=[hetero_secureboost_0.outputs["train_output_data"]]
)

# add training task
pipeline.add_tasks([reader_0, psi_0, hetero_secureboost_0, evaluation_0])

# compile and train
pipeline.compile()
pipeline.fit()

# print metric and model info
print (pipeline.get_task_info("hetero_secureboost_0").get_output_model())
print (pipeline.get_task_info("evaluation_0").get_output_metric())

# save pipeline for later usage
pipeline.dump_model("./pipeline.pkl")

```

5. reload trained pipeline and do prediction
```python
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.components.fate import Reader

# create pipeline for predicting
predict_pipeline = FateFlowPipeline()

# reload trained pipeline
pipeline = FateFlowPipeline.load_model("./pipeline.pkl")

# deploy task for inference
pipeline.deploy([pipeline.psi_0, pipeline.hetero_secureboost_0])

# add input to deployed_pipeline
deployed_pipeline = pipeline.get_deployed_pipeline()
reader_1 = Reader("reader_1")
reader_1.guest.task_parameters(namespace="experiment", name="breast_hetero_guest")
reader_1.hosts[0].task_parameters(namespace="experiment", name="breast_hetero_host")
deployed_pipeline.psi_0.input_data = reader_1.outputs["output_data"]

# add task to predict pipeline
predict_pipeline.add_tasks([reader_1, deployed_pipeline])

# compile and predict
predict_pipeline.compile()
predict_pipeline.predict()
```

6. More tutorials
More pipeline api guides can be found in this [link](https://github.com/FederatedAI/FATE-Client/blob/main/doc/pipeline.md)