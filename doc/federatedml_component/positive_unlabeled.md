# Positive Unlabeled Learning



## Introduction

Positive unlabeled learning is one of the semi-supervised algorithms. The corresponding component is applied to hetero classification tasks of federated learning, in particular, learning classifiers from both positive and unlabeled data.



In the component, unlabeled data are treated as negative data for the binary classifier training. The trained classification model is used for assigning labels to those unlabeled data. After relabeling operations, the dataset is repartitioned and the number of labeled data is increased. Repeatedly, we can gain a model that takes advantage of unlabeled data. The procedure of probability relabeling strategy is shown below.

<div align=center><img src=../images/standard_mode.png/></div>



## Usage
1. Use ***Label Transform*** component to specify the value of unlabeled digit. The unlabeled digit should be set to **0**.

2. Freely connect the combination of ***hetero binary classifier*** and ***Positive Unlabeled*** components in DSL or pipeline.



## Features

Positive unlabeled learning provides different labeling strategies.

* `strategy`: the strategy of converting unlabeled value, including `"probability"`, `"quantity"`, `"proportion"` and `"distribution"`

* `threshold`: the threshold in labeling strategy



## Application

Positive unlabeled learning currently supports the following situations.

* The dataset partition should be positive and unlabeled.

* The model of binary classification can be ***Hetero-LR***, ***Hetero-SSHELR*** or ***Hetero-SecureBoost***.
