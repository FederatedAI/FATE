# Guide to Merging FATE Model

## 1. Overview

Trained FATE models from federated job of `host` and `guest` may be merged locally and reused in other frameworks. 
Currently, FATE supports merging and exporting models from `HeteroLR`, `HeteroSSHELR` and `HeteroSecureBoost` into PMML or sklearn Logistic Regression Model/lightGBM objects.
Below we walk through the general process of merging a Hetero Logistic Regression model using FATE-Flow Client command line tool.

## 2. Install FATE-Flow Client

`flow` command line tool is distributed along with [FATE-Client](https://pypi.org/project/fate-client/).
Once `FATE-Client` installed, one can find a cmd enterpoint named `flow`:

```commandline
pip install fate_client
flow --help
```
```
Usage: flow [OPTIONS] COMMAND [ARGS]...

  Fate Flow Client

Options:
  -h, --help  Show this message and exit.

Commands:
  checkpoint  Checkpoint Operations
  component   Component Operations
  data        Data Operations
  init        Flow CLI Init Command
  job         Job Operations
  key         Key Operations
  model       Model Operations
  privilege   Privilege Operations
  provider    Component Provider Operations
  queue       Queue Operations
  resource    Resource Manager
  server      FATE Flow Server Operations
  service     FATE Flow External Service Operations
  table       Table Operations
  tag         Tag Operations
  task        Task Operations
  template    Template Operations
  test        FATE Flow Test Operations
  tracking    Component Operations
```

To use FATE-Flow Client in terminal, we need to first specify which `FATE Flow Service` to connect to. 
Assume we have a FATE Flow Service at 172.15.0.1:9380, then execute the following command to initialize FATE-Flow Client:

```bash
flow init -c /data/projects/fate/conf/service_conf.yaml
flow init --ip 172.15.0.1 --port 9380
```

In the following steps, we assume that `host` is the role executes merging action.

## 3. Prepare Model

To obtain `guest's` trained Hetero Logistic Regression, we need to first make `host` load exported model.
For models trained in standalone mode, you may skip this step.

### 3.1 Export Model(Cluster mode)

Below is an example model exportation configuration, adopted from [here](https://github.com/FederatedAI/FATE-Flow/blob/main/examples/model/export_model.json):

```json
{
    "role": "guest",
    "party_id": 9999,
    "model_id": "arbiter-10000#guest-9999#host-10000#model",
    "model_version": "202208291446341887230",
    "output_path": "/data/projects/fate/examples/export_model"
}
```

Run the following command to export specified model:

```commandline
flow model export -c export_model.json
``` 

If model exportation succeeds, `flow` will return success message like this:

```
{
    "retcode": 0,
    "file": "/data/projects/fate/examples/model_export/guest#9999#arbiter-10000#guest-9999#host-10000#model_202208291446341887230.zip",
    "retmsg": "download successfully, please check /data/projects/fate/examples/model_export/guest#9999#arbiter-10000#guest-9999#host-10000#model_202208291446341887230.zip"
}
```

### 3.2 Import Model(Cluster mode)

If the FATE model is trained in distributed mode, where `guest` and `host` have FATE each installed in different machines,
exported model needs to be imported/registered on merging role's system first before merging models.

Here is an example configuration for model importation 
adopted from [here](https://github.com/FederatedAI/FATE-Flow/blob/main/examples/model/import_model.json), 
assuming we have put the exported `guest's` model from above step to directory "/data/projects/fate/temp" on `host's` machine:

```json
{
  "role": "guest",
  "party_id": 9999,
  "model_id": "arbiter-10000#guest-9999#host-10000#model",
  "model_version": "202208291446341887230",
  "file": "/data/projects/fate/temp/guest#9999#arbiter-10000#guest-9999#host-10000#model_202208291446341887230.zip"
}
```

Run the following command to import specified model:

```commandline
flow model import -c import_model.json
``` 

If model importation succeeds, `flow` will return success message like this:

```
{
    "data": {
        "job_id": "202208291521250963160",
        "model_id": "arbiter-10000#guest-9999#host-10000#model",
        "model_version": "202208291446341887230",
        "party_id": "9999",
        "role": "guest"
    },
    "retcode": 0,
    "retmsg": "success"
}
```

## 4. Merge Model

### Hetero Logistic Regression Example

Once model is ready, run the following command to merge models into a sklearn Logistic Regression model:

```commandline
flow component hetero-model-merge --model-id arbiter-10000#guest-9999#host-10000#model --model-version 202208291446341887230 --guest-party-id 9999 --host-party-ids 10000 --component-name hetero_lr_0 --model-type lr --output-format sklearn --target-name y --include-guest-coef --output-path model_merge_sklearn
```

We may then load and predict with the obtained sklearn model:

```python
import base64
import json
import pandas
import pickle

filename = "model_merge_sklearn"
test_data_h = pandas.read_csv("/data/projects/fate/examples/data/breast_hetero_host.csv",
                                    header=0, index_col="id")
test_data_g = pandas.read_csv("/data/projects/fate/examples/data/breast_hetero_guest.csv",
                                      header=0, index_col="id")
test_data_g = test_data_g.loc[:, test_data_g.columns != 'y']
test_data = pandas.concat((test_data_g, test_data_h), axis=1)

with open(filename, "r") as f:
    loaded_model = pickle.loads(base64.b64decode(json.load(f)))
    predict_score = loaded_model.predict_proba(test_data)
```

### HeteroSecureBoost Example

If the federated model is a HeteroSecureBoost Model, run this command to merge model and export it in LightGBM format.

```commandline
flow component hetero-model-merge --model-id arbiter-10000#guest-9999#host-10000#model --model-version 202208291446341887230 
--guest-party-id 9999 --host-party-ids 10000 --component-name hetero_secureboost_0 --model-type secureboost --output-format lgb --target-name y --output-path ./lgb.txt
```

If you want a model in PMML format, use this command:

```commandline
flow component hetero-model-merge --model-id arbiter-10000#guest-9999#host-10000#model --model-version 202208291446341887230 
--guest-party-id 9999 --host-party-ids 10000 --component-name hetero_secureboost_0 --model-type secureboost --output-format pmml --target-name y --output-path ./lgb.txt
``` 

Please notice that the model merge function offers a parameter 'host_rename'. Host features will be renamed by adding a sitename
suffix to the original feature name once this option is on:

```commandline
Host party id is 9998, sitename is host_9998, origin feature is x1

x1 -> x1, default setting
x1 -> x1_host_9998, when enable host_rename
```

Use this command to rename host names when output 

```commandline
flow component hetero-model-merge --model-id arbiter-10000#guest-9999#host-10000#model --model-version 202208291446341887230 --host-rename
--guest-party-id 9999 --host-party-ids 10000 --component-name hetero_secureboost_0 --model-type secureboost --output-format lgb --target-name y --output-path ./lgb.txt
```

We may then load and predict with the obtained lightgbm model. Remember that to get the correct predicted result,
the order of features in predicted data should be in the same order as the merged tree model.

```python
import json
import pandas
import lightgbm as lgb

filename = "./lgb.txt"
test_data_h = pandas.read_csv("/data/projects/fate/examples/data/breast_hetero_host.csv",
                                    header=0, index_col="id")
test_data_g = pandas.read_csv("/data/projects/fate/examples/data/breast_hetero_guest.csv",
                                      header=0, index_col="id")
test_data_g = test_data_g.loc[:, test_data_g.columns != 'y']
test_data = pandas.concat((test_data_g, test_data_h), axis=1)

bst = lgb.Booster(model_str=open(filename, 'r').read())

pred_rs = bst.predict(test_data)
```

For more information on command options, please check `help` menu:

```commandline
flow component hetero-model-merge --help
```
```
Usage: flow component hetero-model-merge [OPTIONS]

  - DESCRIPTION:
      Merge a hetero model.

  - USAGE:
      flow component hetero-model-merge --model-id guest-9999#host-9998#model --model-version 202208241838502253290 --guest-party-id 9999 --host-party-ids 9998,9997 --component-name hetero_secure_boost_0 --model-type secureboost --output-format pmml --target-name y --no-host-rename --no-include-guest-coef --output-path model.xml

Options:
  --model-id TEXT                 Model id.  [required]
  --model-version TEXT            Model version.  [required]
  -gid, --guest-party-id TEXT     A valid party id.  [required]
  -hids, --host-party-ids TEXT    Multiple party ids, use a comma to separate
                                  each one.  [required]

  -cpn, --component-name TEXT     A valid component name.  [required]
  --model-type TEXT               [required]
  --output-format TEXT            [required]
  --target-name TEXT
  --host-rename / --no-host-rename
  --include-guest-coef / --no-include-guest-coef
  -o, --output-path PATH          User specifies output directory path.
                                  [required]

  -h, --help                      Show this message and exit.
```


