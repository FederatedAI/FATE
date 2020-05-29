# Data

This document provides explanation and source information on data used for running examples. 
Many of the data sets have been scaled or transformed from their original version.

## Data Set Naming Rule 
All data sets are named according to this guideline: 

table_name: "{content}\_{mode}\_{size}\_{role}\_{role_index}"

- content: brief description of data content
- mode: how original data is divided, either "homo""or hetero"; some data sets do not have this information
- size: includes keyword "mini" if the data set is truncated from another larger set
- role: role name, either "host" or "guest"
- role_index: if a data set is further divided and shared among multiple hosts in some example, 
use indices to distinguish different parties, starts at 1 

Data sets used for running examples are uploaded to local data storage at the time of deployment. 
Uploaded tables share the same `namespace` "experiment" and have `table_name` matching to original file names.
Below lists example data sets and their information. 

## Horizontally Divided Data
> For Homogeneous Federated Learning

#### breast_homo:
- 30 features
- [source](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- data sets: 
    1. "breast_homo_guest.csv"
    2. "breast_homo_host.csv"
    3. "breast_homo_test.csv"

#### default_credit_homo:
- 23 features
- [source](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- data sets: 
    1. "default_credit_homo_guest.csv"
    2. "default_credit_homo_host_1.csv"
    3. "default_credit_homo_host_2.csv"
    4. "default_creidt_homo_test.csv"

#### student_homo:
- 13 features
- [source](https://archive.ics.uci.edu/ml/datasets/student+performance)
- data sets:
    1. "student_homo_guest.csv"
    2. "student_homo_host.csv"
    3. "student_homo_test.csv"

#### vehicle\_scale_homo:
- 18 features
- [source](https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes))
- data sets:
    1. "vehicle_scale_homo_guest.csv"
    2. "vehicle_scale_homo_host.csv"
    3. "vehicle_scale_homo_test.csv"

## Vertically Divided Data
> For Heterogeneous Federated Learning

#### breast_hetero:
- 30 features
- [source](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- data sets:
    1. "breast_hetero_guest.csv"
    2. "breast_hetero_host.csv"
    3. "breast_hetero_mini_guest.csv"

#### breast_hetero_mini:
- 7 features
- [source](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- data sets:
    1. "breast_hetero_mini_guest.csv"
    2. "breast_hetero_mini_host.csv"

#### default_credit_hetero:
- 23 features
- [source](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- data sets:
    1. "default_credit_hetero_guest.csv"
    2. "default_credit_hetero_host.csv"

#### dvisits_hetero:
- 12 features
- [source](https://www.rdocumentation.org/packages/faraway/versions/1.0.7/topics/dvisits)
- data sets:
    1. "dvisits_hetero_guest.csv"
    2. "dvisits_hetero_host.csv"

#### ionosphere_scale_hetero
- 34 features
- [source](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ionosphere_scale)
- data sets:
    1. "ionosphere_scale_hetero_guest.csv"
    2. "ionosphere_scale_hetero_host.csv"

#### motor_hetero:
- 11 features
- [source](https://www.kaggle.com/wkirgsn/electric-motor-temperature)
- data sets:
    1. "motor_hetero_guest.csv"
    2. "motor_hetero_host.csv"

#### motor_hetero_mini:
- 7 features
- [source](https://www.kaggle.com/wkirgsn/electric-motor-temperature)
- data sets:
    1. "motor_hetero_mini_guest.csv"
    2. "motor_hetero_mini_host.csv"
    
#### student_hetero:
- 13 features
- [source](https://archive.ics.uci.edu/ml/datasets/student+performance)
- data sets:
    1. "student_hetero_guest.csv"
    2. "student_hetero_host.csv"

#### vehicle_scale_hetero:
- 18 features
- [source](https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes))
- data sets:
    1. "vehicle_scale_hetero_guest.csv"
    2. "vehicle_scale_hetero_host.csv"
