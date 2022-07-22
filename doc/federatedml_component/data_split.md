# Data Split

Data Split module splits data into desired train, test, and/or validate
sets. The module is based on sklearn train\_test\_split method while its
output can include an extra validate data set.

# Use

Data Split supports homogeneous (both Guest & Host have y) and
heterogeneous (only Guest has y) mode.

Here lists supported split modes and scenario.

| Split Mode 	| Federated Heterogeneous                                                                                            	| Federated Homogeneous(Local)                                                       	|
|------------	|--------------------------------------------------------------------------------------------------------------------	|------------------------------------------------------------------------------------	|
| Random     	| [&check;](../../examples/pipeline/data_split/pipeline-hetero-data-split-multi-model.py)                            	| [&check;](../../examples/pipeline/data_split/pipeline-homo-data-split.py)          	|
| Stratified 	| [&check;](../../examples/pipeline/data_split/pipeline-hetero-data-split.py)(continuous label split into intervals) 	| [&check;](../../examples/pipeline/data_split/pipeline-homo-data-split-validate.py) 	|

The module takes one table input as specified in job config file. Table
must be uploaded beforehand as with other FederatedML models. Module
parameters should be specified in job config file. Any parameter
unspecified will take the default value detailed in
parameter definition below.

Data Split module always outputs three tables (train, test, and validate
sets). Each table may be used as input of another module. Below are the
rules regarding set sizes: 

1. if all three set sizes are None, the
original data input will be split in the following ratio: 80% to train
set, 20% to validate set, and an empty test set; 

2. if only test size or
validate size is given, train size is set to be of complement given
size; 

3. only one of the three sizes is needed to split input data, but
all three may be specified. The module takes either int (instance count)
or float (fraction) value for set sizes, but mixed-type inputs cannot be
used.

<!-- mkdocs
## Param

::: federatedml.param.data_split_param
    rendering:
      heading_level: 3
      show_source: true
      show_root_heading: true
      show_root_toc_entry: false
      show_root_full_path: false
-->

<!-- mkdocs
# Examples

{% include-examples "data_split" %}
-->
