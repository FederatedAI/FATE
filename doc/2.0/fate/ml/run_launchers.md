# Tutorial on Running Launchers¶

### Introduction

In FATE-2.0.0, we introduce launchers for running ml modules locally, a light-weight way to experiment with FATE modules
locally. Running launchers do not require active FATE-Flow services or dependencies from FATE-Flow.

### Installation

Install all requirements of FATE-ML by following command:

```commandline
pip install fate
```

### Create A Launcher

Currently, we provide various ready-to-use launchers for testing mpc protocol and SSHE LR & LinR
modules [here](../../../../launchers).

To write a launcher, first come up with the case to be run with a FATE-module(as in FATE/python/fate/ml) and wrap this
case into a function. As a demo, we are to analyze a simple [launcher](../../../../launchers/sshe_lr_launcher.py) that
trains a SSHE Logistic Regression model using given local data files.

We will use `breast` data set in this demo. Use the following command to download the original data files:

```commandline
wget https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_guest.csv
wget https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_host.csv
```

Local csv data need to be first transformed into DataFrame so that FATE modules may process them. Since this case is a
heterogeneous one, configuration for transformer tool CSVReader will be different for guest and host.


```python
guest_data = 'examples/data/hetero_breast_guest.csv'
host_data = 'examples/data/hetero_breast_host.csv'
if ctx.is_on_guest:
    kwargs = {
        "sample_id_name": None,
        "match_id_name": "id",
        "delimiter": ",",
        "label_name": "y",
        "label_type": "int32",
        "dtype": "float32",
    }
    input_data = dataframe.CSVReader(**kwargs).to_frame(ctx, guest_data)
else:
    kwargs = {
        "sample_id_name": None,
        "match_id_name": "id",
        "delimiter": ",",
        "dtype": "float32",
    }
    input_data = dataframe.CSVReader(**kwargs).to_frame(ctx, host_data)
```

As for the task, first we define a SSEHLR module object, and then feed input data sets into ths module object. At
last, we make this program print out model content.

```python
import logging

logger = logging.getLogger(__name__)


def run_sshe_lr(ctx):
    from fate.ml.glm.hetero.sshe import SSHELogisticRegression
    from fate.arch import dataframe

    ctx.mpc.init()
    inst = SSHELogisticRegression(epochs=5, batch_size=300, tol=0.01, early_stop='diff', learning_rate=0.15,
                                  init_param={"method": "random_uniform", "fit_intercept": True, "random_state": 1},
                                  reveal_every_epoch=False, reveal_loss_freq=2, threshold=0.5)
    inst.fit(ctx, train_data=input_data)
    logger.info(f"model: {pprint.pformat(inst.get_model())}")
```

Combine the above two parts, the program looks like below.

To allow launcher take in user-specified parameters, we also include here argument parser.

```python
import logging
import pprint
from dataclasses import dataclass, field

from fate.arch.launchers.argparser import HfArgumentParser

logger = logging.getLogger(__name__)


@dataclass
class SSHEArguments:
    lr: float = field(default=0.15)
    guest_data: str = field(default=None)
    host_data: str = field(default=None)


def run_sshe_lr(ctx):
    from fate.ml.glm.hetero.sshe import SSHELogisticRegression
    from fate.arch import dataframe

    ctx.mpc.init()
    args, _ = HfArgumentParser(SSHEArguments).parse_args_into_dataclasses(return_remaining_strings=True)
    inst = SSHELogisticRegression(epochs=5, batch_size=300, tol=0.01, early_stop='diff', learning_rate=args.lr,
                                  init_param={"method": "random_uniform", "fit_intercept": True, "random_state": 1},
                                  reveal_every_epoch=False, reveal_loss_freq=2, threshold=0.5)
    if ctx.is_on_guest:
        kwargs = {
            "sample_id_name": None,
            "match_id_name": "id",
            "delimiter": ",",
            "label_name": "y",
            "label_type": "int32",
            "dtype": "float32",
        }
        input_data = dataframe.CSVReader(**kwargs).to_frame(ctx, args.guest_data)
    else:
        kwargs = {
            "sample_id_name": None,
            "match_id_name": "id",
            "delimiter": ",",
            "dtype": "float32",
        }
        input_data = dataframe.CSVReader(**kwargs).to_frame(ctx, args.host_data)
    inst.fit(ctx, train_data=input_data)
    logger.info(f"model: {pprint.pformat(inst.get_model())}")
```

Make sure to use `launch` from `fate.arch` as program entry.

```python
from fate.arch.launchers.multiprocess_launcher import launch
...
if __name__ == "__main__":
    launch(run_sshe_lr, extra_args_desc=[SSHEArguments])
```

### Running A Launcher

As a demo, here we show how to run this Pearson launcher with the following setting from terminal:

- guest: 9999
- host: 10000
- guest_data: examples/data/breast_hetero_guest.csv
- host_data: examples/data/breast_hetero_host.py
- log level: INFO

Note that program will print all logging corresponding to specified log level.

```commandline
python sshe_lr_launcher.py --parties guest:9999 host:10000 --log_level INFO --guest_data breast_hetero_guest.csv --host_data /breast_hetero_host.csv
```

For more launcher examples, please refer [here](../../../../launchers).