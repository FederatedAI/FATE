# Tutorial on Running Launchers

### Introduction

In FATE-2.0.0-rc, we introduce launchers for running ml modules locally, a light-weight way to experiment with FATE
modules locally. Running launchers do not require active FATE-Flow services or dependencies from FATE-Flow.

### Installation

Download FATE-ML code by cloning [FATE repo](https://github.com/FederatedAI/FATE) or downloading zip on the website.

Install all requirements of FATE-ML by following commands if depencies have not been met:

``commandline
pip install -r ${FATE}/python/requiremnts-fate.txt
``

### Create A Launcher

Once dependencies are met, look for launchers directory under FATE.

Currently, we provide various ready-to-use launchers for testing mpc protocol and SSHE LR & LinR modules.

!ls FATE/launchers
To write a launcher, first come up with the case to be run with a FATE-module(as in FATE/python/fate/ml) and wrap this
case into a function. As a demo, we are to analyze a simple [launcher](../../../../launchers/pearson_launcher.py) that
computes pearson correlation coefficient using PearsonCorrelation module from FATE.

First we define a Correlation module object, and then feed input data sets into ths module object to run computation. At
last, we make this program print out local vif values.

```python
def run_pearson(ctx):
    ctx.mpc.init()
    inst = PearsonCorrelation(calc_vif=True)
    ...
    inst.fit(ctx, input_data=input_data)
    print(inst.vif)
```

Local csv data need to be first transformed into DataFrame so that FATE modules may process them. Since our case is a
heterogeneous one, configuration for transformer tool CSVReader will be different for guest and host:

```python
if ctx.is_on_guest:
    kwargs = {
        "sample_id_name": None,
        "match_id_name": "id",
        "delimiter": ",",
        "label_name": "y",
        "label_type": "float32",
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
```

Combine the above two parts, the program looks like this:

```python
def run_pearson(ctx: "Context"):
    from fate.ml.statistics.pearson_correlation import PearsonCorrelation
    from fate.arch import dataframe

    ctx.mpc.init()
    args, _ = HfArgumentParser(PearsonArguments).parse_args_into_dataclasses(return_remaining_strings=True)
    inst = PearsonCorrelation()
    if ctx.is_on_guest:
        kwargs = {
            "sample_id_name": None,
            "match_id_name": "id",
            "delimiter": ",",
            "label_name": "y",
            "label_type": "float32",
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
    inst.fit(ctx, input_data=input_data)
    print(f"role: {ctx.local.role};\n vif: {inst.vif}")
```

Make sure to use `launch` from `fate.arch` as program entry. A launcher generally takes in some user-specified
parameters, you may need to include argument parser in launcher.

```python
from fate.arch.launchers.multiprocess_launcher import launch
...
if __name__ == "__main__":
    launch(run_pearson, extra_args_desc=[PearsonArguments])
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
python FATE/launchers/pearson_launcher.py --parties guest:9999 host:10000 --log_level INFO --guest_data FATE/examples/data/breast_hetero_guest.csv --host_data FATE/examples/data/breast_hetero_host.csv
```

For more launcher examples, please refer [here](../../../../launchers).