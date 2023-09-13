# Feature Scale

Feature scale is a process that scales each feature along column.
Feature Scale module supports min-max scale and standard scale.

1. min-max scale: this estimator scales and translates each feature
   individually such that it is in the given range on the training set,
   e.g. between min and max value of each feature.
2. standard scale: standardize features by removing the mean and
   scaling to unit variance

# Use

| Scale Method | Federated Heterogeneous                                                | 
|--------------|------------------------------------------------------------------------|
| Min-Max      | [&check;](../../../examples/pipeline/sample/test_sample_unilateral.py) | 
| Standard     | [&check;](../../../examples/pipeline/sample/test_sample_unilateral.py) |
