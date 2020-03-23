### Data Split

Data Split module splits data into desired train, test, and/or validate sets. The module is based on sklearn train_test_split method while its output can include an extra validate data set.

### Use

Data Split supports homogeneous (both Guest & Host have y) and heterogeneous (only Guest has y) mode.

The module receives one dtable input as specified in job config file. The data sets must be uploaded beforehand as with other federatedml models. Module parameters may be specified in job config file. Any parameter unspecified will take the default value detailed in [parameter definition](../../param/data_split_param.py).

Data Split module outputs three data sets, and each may be used as input of another module. 

For examples of using Data Split module, please refer [here](../../examples/federatedml-1.x-examples/data_split).