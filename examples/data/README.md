# Data

This document provides explanation and source information on data used for running examples. 
Many of the data sets have been scaled or transformed from their original version.

## Data Set Naming Rule 
All data sets are named according to this guideline: 
 name: "{content}\_{mode}\_{size}\_{role}\_{role_index}"

- content: brief description of data content
- mode: how original data is divided, either "homo""or hetero"; some data sets do not have this information
- size: includes keyword "mini" if the data set is truncated from another larger set
- role: role name, either "host" or "guest"
- role_index: if a data set is further divided and shared among multiple hosts in some example, 
indices are used to distinguish different parties, starts at 1 

Data sets used for running examples are uploaded to FATE data storage at the time of deployment. 
Uploaded tables share the same `namespace` "experiment" and have `table_name` matching to original file names.
Below lists example data sets and their information. 


## Horizontally Divided Data
> For Homogeneous Federated Learning

#### breast_homo:
- 30 features
- label type: binary
- [source](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- data sets: 
    1. "breast_homo_guest.csv"
        * name: "breast_homo_guest"
        * namespace: "experiment"
    2. "breast_homo_host.csv"
        * name: "breast_homo_host"
        * namespace: "experiment"
    3. "breast_homo_test.csv"
        * name: "breast_homo_test"
        * namespace: "experiment"

#### default_credit_homo:
- 23 features
- label type: binary
- [source](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- data sets: 
    1. "default_credit_homo_guest.csv"
        * name: "default_credit_homo_guest"
        * namespace: "experiment"
    2. "default_credit_homo_host_1.csv"
        * name: "default_credit_homo_host_1"
        * namespace: "experiment"
    3. "default_credit_homo_host_2.csv"
        * name: "default_credit_homo_host_2"
        * namespace: "experiment"
    4. "default_credit_homo_test.csv"
        * name: "defeault_credit_homo_test"
        * namespace: "experiment"

#### epsilon_5k_homo:
- 100 features
- label type: binary
- mock data
- data sets:
    1. "epsilon_5k_homo_guest.csv"
        * name: "epsilon_5k_homo_guest"
        * namespace: "experiment"
    2. "epsilon_5k_homo_hostt.csv"
        * name: "epsilon_5k_homo_host"
        * namespace: "experiment"
    3. "epsilon_5k_homo_test.csv"
        * name: "epsilon_5k_homo_test"
        * namespace: "experiment"

#### give_credit_homo:
- 10 features
- label type: binary
- [source](https://www.kaggle.com/c/GiveMeSomeCredit/data)
- data sets:
    1. "give_credit_homo_guest.csv"
        * name: "give_credit_homo_guest"
        * namespace: "experiment"
    2. "give_credit_homo_host.csv"
        * name: "give_credit_homo_host"
        * namespace: "experiment"
    3. "give_credit_homo_test.csv"
        * name: "give_credit_homo_test"
        * namespace: "experiment"

#### student_homo:
- 13 features
- label type: continuous
- [source](https://archive.ics.uci.edu/ml/datasets/student+performance)
- data sets:
    1. "student_homo_guest.csv"
        * name: "student_homo_guest"
        * namespace: "experiment"
    2. "student_homo_host.csv"
        * name: "student_homo_host"
        * namespace: "experiment"
    3. "student_homo_test.csv"
        * name: "student_homo_test"
        * namespace: "experiment"

#### vehicle\_scale_homo:
- 18 features
- label type: multi-class
- [source](https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes))
- data sets:
    1. "vehicle_scale_homo_guest.csv"
        * name: "vehicle_scale_homo_guest"
        * namespace: "experiment"
    2. "vehicle_scale_homo_host.csv"
        * name: "vehicle_scale_homo_host"
        * namespace: "experiment"
    3. "vehicle_scale_homo_test.csv"
        * name: "vehicle_scale_homo_test"
        * namespace: "experiment"

## Vertically Divided Data
> For Heterogeneous Federated Learning

#### breast_hetero:
- 30 features
- label type: binary
- [source](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- data sets:
    1. "breast_hetero_guest.csv"
        * name: "breast_hetero_guest"
        * namespace: "experiment"
    2. "breast_hetero_host.csv"
        * name: "breast_hetero_host"
        * namespace: "experiment"

#### breast_hetero_mini:
- 7 features
- label type: binary
- [source](https://www.kaggle.com/uciml/breast-cancer-wisconsin-data)
- data sets:
    1. "breast_hetero_mini_guest.csv"
        * name: "breast_hetero_mini_guest"
        * namespace: "experiment"
    2. "breast_hetero_mini_host.csv"
        * name: "breast_hetero_mini_host"
        * namespace: "experiment"

#### default_credit_hetero:
- 23 features
- label type: binary
- [source](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- data sets:
    1. "default_credit_hetero_guest.csv"
        * name: "default_credit_hetero_guest"
        * namespace: "experiment"
    2. "default_credit_hetero_host.csv"
        * name: "default_credit_hetero_host"
        * namespace: "experiment"

#### dvisits_hetero:
- 12 features
- label type: continuous
- [source](https://www.rdocumentation.org/packages/faraway/versions/1.0.7/topics/dvisits)
- data sets:
    1. "dvisits_hetero_guest.csv"
        * name: "dvisits_hetero_guest"
        * namespace: "experiment"
    2. "dvisits_hetero_host.csv"
        * name: "dvisits_hetero_host"
        * namespace: "experiment"

#### epsilon_5k_hetero:
- 100 features
- label type: binary
- mock data
- data sets:
    1. "epsilon_5k_hetero_guest.csv"
        * name: "epsilon_5k_hetero_guest"
        * namespace: "experiment"
    2. "epsilon_5k_hetero_host.csv"
        * name: "epsilon_5k_hetero_host"
        * namespace: "experiment"

#### give_credit_hetero:
- 10 features
- label type: binary
- [source](https://www.kaggle.com/c/GiveMeSomeCredit/data)
- data sets:
    1. "give_credit_hetero_guest.csv"
        * name: "give_credit_hetero_guest"
        * namespace: "experiment"
    2. "give_credit_hetero_host.csv"
        * name: "give_credit_hetero_host"
        * namespace: "experiment"

#### ionosphere_scale_hetero
- 34 features
- label type: binary
- [source](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ionosphere_scale)
- data sets:
    1. "ionosphere_scale_hetero_guest.csv"
        * name: "ionosphere_scale_hetero_guest"
        * namespace: "experiment"
    2. "ionosphere_scale_hetero_host.csv"
        * name: "ionosphere_scale_hetero_host"
        * namespace: "experiment"

#### motor_hetero:
- 11 features
- label type: continuous
- [source](https://www.kaggle.com/wkirgsn/electric-motor-temperature)
- data sets:
    1. "motor_hetero_guest.csv"
        * name: "motor_hetero_guest"
        * namespace: "experiment"
    2. "motor_hetero_host.csv"
        * name: "motor_hetero_host"
        * namespace: "experiment"
    3. "motor_hetero_host_1.csv"
        * name: "motor_hetero_host_1"
        * namespace: "experiment"
    4. "motor_hetero_host_2.csv"
        * name: "motor_hetero_host_2"
        * namespace: "experiment"

#### motor_hetero_mini:
- 7 features
- label type: continuous
- [source](https://www.kaggle.com/wkirgsn/electric-motor-temperature)
- data sets:
    1. "motor_hetero_mini_guest.csv"
        * name: "motor_hetero_mini_guest"
        * namespace: "experiment"
    2. "motor_hetero_mini_host.csv"
        * name: "motor_hetero_mini_host"
        * namespace: "experiment"
    
#### student_hetero:
- 13 features
- label type: continuous
- [source](https://archive.ics.uci.edu/ml/datasets/student+performance)
- data sets:
    1. "student_hetero_guest.csv"
        * name: "student_hetero_guest"
        * namespace: "experiment"
    2. "student_hetero_host.csv"
        * name: "student_hetero_host"
        * namespace: "experiment"

#### vehicle_scale_hetero:
- 18 features
- label type: multi-class
- [source](https://archive.ics.uci.edu/ml/datasets/Statlog+(Vehicle+Silhouettes))
- data sets:
    1. "vehicle_scale_hetero_guest.csv"
        * name: "vehicle_scale_hetero_guest"
        * namespace: "experiment"
    2. "vehicle_scale_hetero_host.csv"
        * name: "vehicle_scale_hetero_host"
        * namespace: "experiment"


## Federated Transfer Learning Data
> For Federated Transfer Learning

### nus_wide:
- 636/1000 features
- label type: binary
- [source](https://lms.comp.nus.edu.sg/wp-content/uploads/2019/research/nuswide/NUS-WIDE.html)
- data sets:
    1. "nus_wide_train_guest.csv"
        * name: "nus_wide_train_guest"
        * namespace: "experiment"
    2. "nus_wide_train_host.csv"
        * name: "nus_wide_train_host"
        * namespace: "experiment"
    3. "nus_wide_validate_guest.csv"
        * name: "nus_wide_validate_guest"
        * namespace: "experiment"
    4. "nus_wide_validate_guest.csv"
        * name: "nus_wide_validate_guest"
        * namespace: "experiment"


## Non-Divided Data
> Generated Data for Data Operation Demo

#### tag_value:
- data sets:
    1. "tag_value_1000_140.csv"
        * name: "tag_value_1", "tag_value_2", "tag_value_3"
        * namespace: "experiment"
