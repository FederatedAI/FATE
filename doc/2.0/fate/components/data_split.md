# Data Split

Data Split module splits data into train, test, and/or validate
sets of arbitrary sizes. The module is based on sampling method.

# Use

Data Split supports local(same as homogeneous) and heterogeneous (only Guest has y) mode.

Here lists supported split modes and scenario.

| Split Mode 	 | Federated Heterogeneous                                                        | Federated Homogeneous(Local)                                                   |
|--------------|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| Random     	 | [&check;](../../../examples/pipeline/data_split/test_data_split.py)            | [&check;](../../../examples/pipeline/data_split/test_data_split_multi_host.py) |
| Stratified 	 | [&check;](../../../examples/pipeline/data_split/test_data_split_stratified.py) | [&check;](../../../examples/pipeline/data_split/test_data_split_stratified.py) |

Data Split module takes single data input as specified in job config file
and always outputs three tables (train, test, and validate
data sets). Each data ouput may be used as input of another module. Below are the
rules regarding set sizes:

1. if all three set sizes are None, the
   original data input will be split in the following ratio: 80% to train
   set, 20% to validate set, and an empty test set;

2. if only test size or
   validate size is given, train size is set to be of complement given
   size;

3. only one of the three sizes is needed to split input data, but
   all three may be specified. The module takes either int (instance count)
   or float (fraction) value for set sizes, but mixed-type inputs are not accepted.
