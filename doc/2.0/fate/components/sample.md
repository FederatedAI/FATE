# Federated Sampling

Sample module supports random sampling and stratified sampling.

- `hetero_sync` should be set to True for heterogeneous scenario;
- `replace` must be set to True if upsample is needed.

| Sample Type 	     | Federated Heterogeneous                                     | Federated Homogeneous(Local)                                                   |
|-------------------|-------------------------------------------------------------|--------------------------------------------------------------------------------|
| By Fraction     	 | [&check;](../../../../examples/pipeline/sample/test_sample.py) | [&check;](../../../../examples/pipeline/sample/test_data_split_multi_host.py)     |
| By Exact Number 	 | [&check;](../../../../examples/pipeline/sample/test_sample.py) | [&check;](../../../../examples/pipeline/data_split/test_data_split_stratified.py) |
