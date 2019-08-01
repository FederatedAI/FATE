## Components in FATE

Currently, FATE support these following components. This list will keep updating when new components added.

Each component has a strictly corresponding module name which will be used in dsl conf.

```
    "dataio_0": {
        "module": "DataIO",
        ...
        }
```

### 1. Data-io
This component is typically the first component of a modeling task. It will transform user-uploaded date into DTable which can be used for the following components.

Corresponding module name: DataIO

### 2. Intersect
Compute intersect data set of two parties without leakage of difference set information. Mainly used in hetero scenario task.

Corresponding module name: Intersection

### 3. Federated Sampling
Federated Sampling data so that its distribution become balance in each party.This module support both federated and standalone version

Corresponding module name: FederatedSample

### 4. Feature Scale
Module for feature scaling and standardization.

Corresponding module name: FeatureScale

### 5. Hetero Feature Binning
With binning input data, calculates each column's iv and woe and transform data according to the binned information.

Corresponding module name: HeteroFeatureBinning

### 6. OneHot Encoder
Transfer a column into one-hot format.

Corresponding module name: OneHotEncoder

### 7. Hetero Feature Selection
Provide 5 types of filters. Each filters can select columns according to user config.

Corresponding module name: HeteroFeatureSelection

### 8. Hetero-LR
Build hetero logistic regression module through multiple parties.

Corresponding module name: HeteroLR

### 9. Homo-LR
Build homo logistic regression module through multiple parties.

Corresponding module name: HomoLR

### 10. Hetero Secure Boosting
Build hetero secure boosting model through multiple parties.

Corresponding module name: HeteroSecureBoost

### 11. Evaluation
Output the model evaluation metrics for user.

Corresponding module name: Evaluation
