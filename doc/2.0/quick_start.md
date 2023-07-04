## Quick Start

1. install `fate_client` with extra package `fate`  

```sh
python -m pip install -U pip && python -m pip install fate_client[fate]==2.0.0a0
```

2.  download example data

```sh
wget https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_guest.csv && \
wget https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_host.csv
```

3. run example with fate_client

```python
import os

from fate_client.pipeline import StandalonePipeline
from fate_client.pipeline.components.fate import (
    Evaluation,
    FeatureScale,
    HeteroLR,
    Intersection,
    Reader,
)

base_path = os.path.abspath(os.path.join(__file__, os.path.pardir))
guest_data_path = os.path.join(base_path, "breast_hetero_guest.csv")
host_data_path = os.path.join(base_path, "breast_hetero_host.csv")

# create pipeline
pipeline = StandalonePipeline().set_roles(guest="9999", host="10000", arbiter="10001")

# create reader component_desc
reader_0 = Reader(name="reader_0")
reader_0.guest.component_param(
    path=f"file://${guest_data_path}",
    format="csv",
    id_name="id",
    delimiter=",",
    label_name="y",
    label_type="float32",
    dtype="float32",
)
reader_0.hosts[0].component_param(
    path=f"file://${host_data_path}",
    format="csv",
    id_name="id",
    delimiter=",",
    label_name=None,
    dtype="float32",
)

# create intersection component_desc
intersection_0 = Intersection(name="intersection_0", method="raw", input_data=reader_0.outputs["output_data"])
intersection_1 = Intersection(name="intersection_1", method="raw", input_data=reader_0.outputs["output_data"])

# create feature scale component_desc
feature_scale_0 = FeatureScale(
    name="feature_scale_0", method="standard", train_data=intersection_0.outputs["output_data"]
)
feature_scale_1 = FeatureScale(
    name="feature_scale_1",
    test_data=intersection_1.outputs["output_data"],
    input_model=feature_scale_0.outputs["output_model"],
)

# create lr component_desc
lr_0 = HeteroLR(
    name="lr_0",
    train_data=feature_scale_0.outputs["train_output_data"],
    validate_data=feature_scale_1.outputs["test_output_data"],
    max_iter=100,
    learning_rate=0.03,
    batch_size=-1,
)

# create evaluation component_desc
evaluation_0 = Evaluation(name="evaluation_0", runtime_roles="guest", input_data=lr_0.outputs["train_output_data"])

# add components
pipeline.add_task(reader_0)
pipeline.add_task(feature_scale_0)
pipeline.add_task(feature_scale_1)
pipeline.add_task(intersection_0)
pipeline.add_task(intersection_1)
pipeline.add_task(lr_0)
pipeline.add_task(evaluation_0)

# train
pipeline.compile()
print(pipeline.get_dag())
pipeline.fit()
print(pipeline.get_task_info("feature_scale_0").get_output_model())
print(pipeline.get_task_info("lr_0").get_output_model())
print(pipeline.get_task_info("lr_0").get_output_data())
print(pipeline.get_task_info("evaluation_0").get_output_metrics())
print(pipeline.deploy([intersection_0, feature_scale_0, lr_0]))
```
