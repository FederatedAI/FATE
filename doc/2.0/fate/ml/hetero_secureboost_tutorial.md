# Hetero-SecureBoost Tutorial

In a hetero-federated learning (vertically partitioned data) setting, multiple parties have different feature sets for the same common user samples. Federated learning enables these parties to collaboratively train a model without sharing their actual data. The model is trained locally at each party, and only model updates are shared, not the actual data. 
SecureBoost is a specialized tree-boosting framework designed for vertical federated learning. It performs entity alignment under a privacy-preserving protocol and constructs boosting trees across multiple parties using an encryption strategy. It allows for high-quality, lossless model training without needing a trusted third party.

In this tutorial, we will show you how to run a Hetero-SecureBoost task under FATE-2.0 locally without using a FATE-Pipeline. You can refer to this example for local model experimentation, algorithm modification, and testing.

## Setup Hetero-Secureboost Step by Step

To run a Hetero-Secureboost task, several steps are needed:
1. Import required classes in a new python script
2. Prepare tabular data and transform them into fate dataframe
3. Create Launch Function and Run Hetero-Secureboost task


## Import Libs and Write a Python Script
We import these classes for later use.

```python
import pandas as pd
from fate.arch.dataframe import PandasReader
from fate.ml.ensemble.algo.secureboost.hetero.guest import HeteroSecureBoostGuest
from fate.ml.ensemble.algo.secureboost.hetero.host import HeteroSecureBoostHost
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from datetime import datetime
from fate.arch.context import create_context
```

### Fate Context

FATE-2.0 uses a context object to configure the running environment, including party setting(guest, host and theirs party ids). We can create a context object by calling the create_context function.

```python
def create_ctx(party, session_id='test_fate'):
    parties = [("guest", "9999"), ("host", "10000")]
    if party == "guest":
        local_party = ("guest", "9999")
    else:
        local_party = ("host", "10000")
    context = create_context(local_party, parties=parties, federation_session_id=session_id)
    return context
```

If we run our Hetero-Secureboost task with launch(), it can automatically handle the context creation, this chapter will show you how to do it manually.
In the data preparation step, if you want to check your data before training, you can create context manually and check your data in FATE-DataFrame 


### Train and Predict Functions

First we define a train function which initializes a HeteroSecureBoostingTree object for guest/host party and calls the fit method to train the model. This function takes a ctx object, FATE Dataframe instances, and some parameters as input. The ctx object is used to configure the running environment, while the FATE Dataframe instances are used to store the training data. We can specify training parameters in the fit method, check out the HeteroSecureBoostGuest and HeteroSecureBoostHost classes for more details. 

```python
def train(ctx: Context, data: DataFrame, num_trees: int = 3, objective: str = 'binary:bce', max_depth: int = 3, learning_rate: float=0.3):
    
    if ctx.is_on_guest:
        bst = HeteroSecureBoostGuest(num_trees=num_trees, objective=objective, \
            max_depth=max_depth, learning_rate=learning_rate)
    else:
        bst = HeteroSecureBoostHost(num_trees=num_trees, max_depth=max_depth)

    bst.fit(ctx, data)

    return bst
```

After the training process is done, we can call the get_model method to get the trained model. We can get the model dict using get_model() and load a new tree model using from_model().

```python
model_dict = bst.get_model()
# take guest side as an example
bst_2 = HeteroSecureBoostGuest()
bst_2.from_model(model_dict)
```

We can predict new data with trained model. Here we define a predict function which takes a ctx object, FATE Dataframe instances, and model dict as input. 

```python
def predict(ctx: Context, data: DataFrame, model_dict: dict):
    ctx = ctx.sub_ctx('predict')
    if ctx.is_on_guest:
        bst = HeteroSecureBoostGuest()
    else:
        bst = HeteroSecureBoostHost()
    bst.from_model(model_dict)
    return bst.predict(ctx, data)
```


## Prepare Tabular Data and Transform into FATE-DataFrame

You can download our example data from: 

- [breast_hetero_guest.csv](https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_guest.csv)
- [breast_hetero_host.csv](https://raw.githubusercontent.com/wiki/FederatedAI/FATE/example/data/breast_hetero_host.csv)


And put them in the same directory as this tutorial.
Here we write a function that reads data from a csv file and transforms it into a FATE-DataFrame. Please notice that in the hetero-federated learning setting, the guest party has label data while the host party does not. So we need to specify the label column when reading data for the guest party. Then we will verify the function with our example data. 


```python
def csv_to_df(ctx, file_path, has_label=True):

    df = pd.read_csv(file_path)
    df["sample_id"] = [i for i in range(len(df))]
    if has_label:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", label_name="y", dtype="float32") 
    else:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", dtype="float32")

    fate_df = reader.to_frame(ctx, df)
    return fate_df
```

Let us check the data we just download.

```python
ctx = create_ctx('guest')
data = csv_to_df(ctx, './breast_hetero_guest.csv')
print('guest_data')
print(data.as_pd_df().head())
```

```python
ctx = create_ctx('host')
data = csv_to_df(ctx, './breast_hetero_host.csv', has_label=False)
print('host_data')
print(data.as_pd_df().head())
```

Here is the console output:

```console
guest_data
  sample_id     id  y        x0        x1        x2        x3        x4        x5        x6        x7        x8        x9
0         0  133.0  1  0.254879 -1.046633  0.209656  0.074214 -0.441366 -0.377645 -0.485934  0.347072 -0.287570 -0.733474
1         5  274.0  0  0.963102  1.467675  0.829202  0.772457 -0.038076 -0.468613 -0.307946 -0.015321 -0.641864 -0.247477
2         6  420.0  1 -0.662496  0.212149 -0.620475 -0.632995 -0.327392 -0.385278 -0.077665 -0.730362  0.217178 -0.061280
3         7   76.0  1 -0.453343 -2.147457 -0.473631 -0.483572  0.558093 -0.740244 -0.896170 -0.617229 -0.308601 -0.666975
4         8  315.0  1 -0.606584 -0.971725 -0.678558 -0.591332 -0.963013 -1.302401 -1.212855 -1.321154 -1.591501 -1.230554
```

```consle
host_data   
sample_id     id        x0        x1        x2        x3        x4        x5        x6        x7        x8        x9       x10       x11       x12       x13       x14       x15       x16       x17       x18       x19
0         0  133.0  0.449512 -1.247226  0.413178  0.303781 -0.123848 -0.184227 -0.219076  0.268537  0.015996 -0.789267 -0.337360 -0.728193 -0.442587 -0.272757 -0.608018 -0.577235 -0.501126  0.143371 -0.466431 -0.554102
1         5  274.0  1.080023  1.207830  0.956888  0.978402 -0.555822 -0.645696 -0.399365 -0.038153 -0.998966 -1.091216  0.057848  0.392164 -0.050027  0.120414 -0.532348 -0.770613 -0.519694 -0.531097 -0.769127 -0.394858
2         6  420.0 -0.726307 -0.058095 -0.731910 -0.697343 -0.775723 -0.513983 -0.426233 -0.893482  0.800949 -0.018090 -0.428673  0.404865 -0.326750 -0.440850  0.079010 -0.279903  0.416992 -0.486165 -0.225484 -0.172446
3         7   76.0 -0.169639 -1.943019 -0.167192 -0.272150  2.329937  0.006804 -0.251467  0.429234  2.159100  0.512094  0.017786 -0.368046 -0.105966 -0.169129  2.119760  0.162743 -0.672216 -0.577002  0.626908  0.896114
4         8  315.0 -0.465014 -0.567723 -0.526371 -0.492852 -0.800631 -1.250816 -1.058714 -1.096145 -2.178221 -0.860147 -0.843011 -0.910353 -0.900490 -0.608283 -0.704355 -1.255622 -0.970629 -1.363557 -0.800607 -0.927058
```


## Launch Hetero-Secureboost Task with launch()

The launch can automatically handle the context creation and simulate a local federation learning, so we can easily run task with a single command. The last thing we need to do is to write a running function which takes a ctx object as input.

```python
from fate.arch.launchers.multiprocess_launcher import launch

def run(ctx):
    num_tree = 3
    max_depth = 3
    if ctx.is_on_guest:
        data = csv_to_df(ctx, './breast_hetero_guest.csv')
        bst = train(ctx, data, num_trees=num_tree, max_depth=max_depth)
        model_dict = bst.get_model()
        pred = predict(ctx, data, model_dict)
        print(pred.as_pd_df())
    else:
        data = csv_to_df(ctx, './breast_hetero_host.csv', has_label=False)
        bst = train(ctx, data, num_trees=num_tree, max_depth=max_depth)
        model_dict = bst.get_model()
        predict(ctx, data, model_dict)

if __name__ == '__main__':
    launch(run)
```

Below is the full script.

### Full Script 
```python
import pandas as pd
from fate.arch.dataframe import PandasReader
from fate.ml.ensemble.algo.secureboost.hetero.guest import HeteroSecureBoostGuest
from fate.ml.ensemble.algo.secureboost.hetero.host import HeteroSecureBoostHost
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from datetime import datetime
from fate.arch.context import create_context
from fate.arch.launchers.multiprocess_launcher import launch


def train(ctx: Context, data: DataFrame, num_trees: int = 3, objective: str = 'binary:bce', max_depth: int = 3, learning_rate: float=0.3):
    
    if ctx.is_on_guest:
        bst = HeteroSecureBoostGuest(num_trees=num_trees, objective=objective, \
            max_depth=max_depth, learning_rate=learning_rate)
    else:
        bst = HeteroSecureBoostHost(num_trees=num_trees, max_depth=max_depth)

    bst.fit(ctx, data)

    return bst

def predict(ctx: Context, data: DataFrame, model_dict: dict):
    if ctx.is_on_guest:
        bst = HeteroSecureBoostGuest()
    else:
        bst = HeteroSecureBoostHost()
    bst.from_model(model_dict)
    return bst.predict(ctx, data)


def csv_to_df(ctx, file_path, has_label=True):

    df = pd.read_csv(file_path)
    df["sample_id"] = [i for i in range(len(df))]
    if has_label:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", label_name="y", dtype="float32") 
    else:
        reader = PandasReader(sample_id_name="sample_id", match_id_name="id", dtype="float32")

    fate_df = reader.to_frame(ctx, df)
    return fate_df

def run(ctx):
    num_tree = 3
    max_depth = 3
    if ctx.is_on_guest:
        data = csv_to_df(ctx, './breast_hetero_guest.csv')
        bst = train(ctx, data, num_trees=num_tree, max_depth=max_depth)
        model_dict = bst.get_model()
        pred = predict(ctx, data, model_dict)
        print(pred.as_pd_df())
    else:
        data = csv_to_df(ctx, './breast_hetero_host.csv', has_label=False)
        bst = train(ctx, data, num_trees=num_tree, max_depth=max_depth)
        model_dict = bst.get_model()
        predict(ctx, data, model_dict)

if __name__ == '__main__':
    launch(run)
```

We can run this script 
```
python sbt.py --parties guest:9999 host:10000 --log_level INFO
```

Here is the terminal output:

```console
[16:13:01] INFO     [ Main ] ========================================================                                                                                                                                                                         multiprocess_launcher.py:277
           INFO     [ Main ] federation id: 20231227161301-c09d1c                                                                                                                                                                                             multiprocess_launcher.py:278
           INFO     [ Main ] parties: ['guest:9999', 'host:10000']                                                                                                                                                                                            multiprocess_launcher.py:279
           INFO     [ Main ] data dir: None                                                                                                                                                                                                                   multiprocess_launcher.py:280
           INFO     [ Main ] ========================================================                                                                                                                                                                         multiprocess_launcher.py:281
           INFO     [ Main ] disabled tracing                                                                                                                                                                                                                                 _trace.py:31
           INFO     [ Main ] waiting for all processes to exit                                                                                                                                                                                                multiprocess_launcher.py:220
[16:13:02] INFO     [Rank:0] disabled tracing                                                                                                                                                                                                                                 _trace.py:31
[16:13:02] INFO     [Rank:1] disabled tracing                                                                                                                                                                                                                                 _trace.py:31
[16:13:06] INFO     [Rank:1] data binning done                                                                                                                                                                                                                                  host.py:66
[16:13:07] INFO     [Rank:1] tree dimension is 1                                                                                                                                                                                                                                host.py:59
[16:13:07] INFO     [Rank:0] start to fit a guest tree                                                                                                                                                                                                                        guest.py:286
           INFO     [Rank:0] encrypt kit setup through setter                                                                                                                                                                                                                 guest.py:129
[16:13:08] INFO     [Rank:0] gh are packed                                                                                                                                                                                                                                    guest.py:238
[16:13:09] INFO     [Rank:1] cur layer node num: 1, next layer node num: 2                                                                                                                                                                                                     host.py:205
[16:13:11] INFO     [Rank:1] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] layer 0 done: next layer will split 2 nodes, active samples num 569                                                                                                                                                                               host.py:214
[16:13:11] INFO     [Rank:0] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:0] layer 0 done: next layer will split 2 nodes, active samples num 569                                                                                                                                                                              guest.py:387
[16:13:12] INFO     [Rank:1] cur layer node num: 2, next layer node num: 4                                                                                                                                                                                                     host.py:205
[16:13:13] INFO     [Rank:1] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] layer 1 done: next layer will split 4 nodes, active samples num 569                                                                                                                                                                               host.py:214
[16:13:14] INFO     [Rank:0] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:0] layer 1 done: next layer will split 4 nodes, active samples num 569                                                                                                                                                                              guest.py:387
[16:13:15] INFO     [Rank:1] cur layer node num: 4, next layer node num: 8                                                                                                                                                                                                     host.py:205
[16:13:16] INFO     [Rank:1] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] layer 2 done: next layer will split 8 nodes, active samples num 569                                                                                                                                                                               host.py:214
[16:13:16] INFO     [Rank:0] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] fitting host decision tree 0, dim 0 done                                                                                                                                                                                                           host.py:94
           INFO     [Rank:0] layer 2 done: next layer will split 8 nodes, active samples num 569                                                                                                                                                                              guest.py:387
[16:13:17] INFO     [Rank:0] fitting guest decision tree iter 0, dim 0 done                                                                                                                                                                                                   guest.py:325
[16:13:19] INFO     [Rank:0] start to fit a guest tree                                                                                                                                                                                                                        guest.py:286
           INFO     [Rank:0] encrypt kit setup through setter                                                                                                                                                                                                                 guest.py:129
[16:13:20] INFO     [Rank:0] gh are packed                                                                                                                                                                                                                                    guest.py:238
[16:13:21] INFO     [Rank:1] cur layer node num: 1, next layer node num: 2                                                                                                                                                                                                     host.py:205
[16:13:22] INFO     [Rank:1] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] layer 0 done: next layer will split 2 nodes, active samples num 569                                                                                                                                                                               host.py:214
[16:13:22] INFO     [Rank:0] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:0] layer 0 done: next layer will split 2 nodes, active samples num 569                                                                                                                                                                              guest.py:387
[16:13:23] INFO     [Rank:1] cur layer node num: 2, next layer node num: 4                                                                                                                                                                                                     host.py:205
[16:13:25] INFO     [Rank:1] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] layer 1 done: next layer will split 4 nodes, active samples num 569                                                                                                                                                                               host.py:214
[16:13:25] INFO     [Rank:0] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:0] layer 1 done: next layer will split 4 nodes, active samples num 569                                                                                                                                                                              guest.py:387
[16:13:26] INFO     [Rank:1] cur layer node num: 4, next layer node num: 8                                                                                                                                                                                                     host.py:205
[16:13:28] INFO     [Rank:1] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] layer 2 done: next layer will split 8 nodes, active samples num 569                                                                                                                                                                               host.py:214
[16:13:28] INFO     [Rank:0] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] fitting host decision tree 1, dim 0 done                                                                                                                                                                                                           host.py:94
           INFO     [Rank:0] layer 2 done: next layer will split 8 nodes, active samples num 569                                                                                                                                                                              guest.py:387
[16:13:29] INFO     [Rank:0] fitting guest decision tree iter 1, dim 0 done                                                                                                                                                                                                   guest.py:325
[16:13:30] INFO     [Rank:0] start to fit a guest tree                                                                                                                                                                                                                        guest.py:286
           INFO     [Rank:0] encrypt kit setup through setter                                                                                                                                                                                                                 guest.py:129
[16:13:31] INFO     [Rank:0] gh are packed                                                                                                                                                                                                                                    guest.py:238
[16:13:32] INFO     [Rank:1] cur layer node num: 1, next layer node num: 2                                                                                                                                                                                                     host.py:205
[16:13:34] INFO     [Rank:1] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] layer 0 done: next layer will split 2 nodes, active samples num 569                                                                                                                                                                               host.py:214
[16:13:34] INFO     [Rank:0] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:0] layer 0 done: next layer will split 2 nodes, active samples num 569                                                                                                                                                                              guest.py:387
[16:13:35] INFO     [Rank:1] cur layer node num: 2, next layer node num: 4                                                                                                                                                                                                     host.py:205
[16:13:36] INFO     [Rank:1] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
[16:13:37] INFO     [Rank:1] layer 1 done: next layer will split 4 nodes, active samples num 569                                                                                                                                                                               host.py:214
[16:13:37] INFO     [Rank:0] drop leaf samples, new sample count is 569, 0 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:0] layer 1 done: next layer will split 4 nodes, active samples num 569                                                                                                                                                                              guest.py:387
[16:13:38] INFO     [Rank:0] Node 4 can not be further split                                                                                                                                                                                                               splitter.py:248
           INFO     [Rank:0] Node 4 can not be further split                                                                                                                                                                                                               splitter.py:248
[16:13:38] INFO     [Rank:1] cur layer node num: 4, next layer node num: 6                                                                                                                                                                                                     host.py:205
[16:13:39] INFO     [Rank:1] drop leaf samples, new sample count is 569, 5 samples dropped                                                                                                                                                                            decision_tree.py:420
[16:13:40] INFO     [Rank:1] layer 2 done: next layer will split 6 nodes, active samples num 564                                                                                                                                                                               host.py:214
[16:13:40] INFO     [Rank:0] drop leaf samples, new sample count is 569, 5 samples dropped                                                                                                                                                                            decision_tree.py:420
           INFO     [Rank:1] fitting host decision tree 2, dim 0 done                                                                                                                                                                                                           host.py:94
           INFO     [Rank:0] layer 2 done: next layer will split 6 nodes, active samples num 564                                                                                                                                                                              guest.py:387
[16:13:41] INFO     [Rank:0] fitting guest decision tree iter 2, dim 0 done                                                                                                                                                                                                   guest.py:325
[16:13:43] INFO     [Rank:0] predict round 0 has 569 samples to predict                                                                                                                                                                                                     predict.py:138
[16:13:44] INFO     [Rank:1] got 564 pending samples                                                                                                                                                                                                                        predict.py:189
[16:13:46] INFO     [Rank:0] predict round 1 has 564 samples to predict                                                                                                                                                                                                     predict.py:138
[16:13:47] INFO     [Rank:0] predict done                                                                                                                                                                                                                                   predict.py:166
[16:13:47] INFO     [Rank:1] predict done                                                                                                                                                                                                                                   predict.py:197
[16:13:48] INFO     [Rank:1] Total: 45.6226s, Driver: 11.1416s(24.42%), Federation: 19.6592s(43.09%), Computing: 14.8219s(32.49%)                                                                                                                                          _profile.py:279
           INFO     [Rank:1]                                                                                                                                                                                                                                               _profile.py:290
                    Computing:                                                                                                                                                                                                                                                            
                    +----------+---------------------------------------------------------+                                                                                                                                                                                                
                    | function |       function          n    sum(s)   mean(s)   max(s)  |                                                                                                                                                                                                
                    |          | --------------------- ----- -------- --------- -------- |                                                                                                                                                                                                
                    |          |       mapValues        106   4.363     0.041    0.198   |                                                                                                                                                                                                
                    |          |    applyPartitions     20    4.242     0.212    3.627   |                                                                                                                                                                                                
                    |          |         join           63    2.673     0.042    0.091   |                                                                                                                                                                                                
                    |          |  mapReducePartitions   10    2.085     0.208    0.236   |                                                                                                                                                                                                
                    |          |     mapPartitions      12    1.244     0.104    0.124   |                                                                                                                                                                                                
                    |          |         first          19    0.114     0.006    0.009   |                                                                                                                                                                                                
                    |          |      parallelize        1    0.047     0.047    0.047   |                                                                                                                                                                                                
                    |          |         take           19    0.035     0.002    0.003   |                                                                                                                                                                                                
                    |          |        reduce           2    0.011     0.005    0.007   |                                                                                                                                                                                                
                    |          |   repartition_with     63    0.006      0.0      0.0    |                                                                                                                                                                                                
                    |          |        collect         19    0.002      0.0      0.0    |                                                                                                                                                                                                
                    |          |         count           1     0.0       0.0      0.0    |                                                                                                                                                                                                
                    +----------+---------------------------------------------------------+                                                                                                                                                                                                
                    |  total   |       n=335, sum=14.8219, mean=0.0442, max=3.6273       |                                                                                                                                                                                                
                    +----------+---------------------------------------------------------+                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                          
                    Federation:                                                                                                                                                                                                                                                           
                    +--------+---------------------------------------------------+                                                                                                                                                                                                        
                    |  get   |       name       | n | sum(s) | mean(s) | max(s)  |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |       en_gh      | 3 | 7.537  |  2.512  | 3.089   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |     need_stop    | 2 | 6.259  |  3.13   | 3.919   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |  new_sample_pos  | 9 | 2.644  |  0.294  | 0.308   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |    sync_nodes    | 9 | 1.518  |  0.169  | 0.203   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |     tree_dim     | 1 | 0.593  |  0.593  | 0.593   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |   updated_data   | 9 |  0.32  |  0.036  | 0.041   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |       hist       | 9 |  0.31  |  0.034  | 0.039   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |  pending_samples | 1 | 0.287  |  0.287  | 0.287   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |     pack_info    | 3 | 0.121  |  0.04   | 0.042   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |      en_kit      | 3 | 0.034  |  0.011  | 0.013   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |    updated_pos   | 1 | 0.034  |  0.034  | 0.034   |                                                                                                                                                                                                        
                    +--------+---------------------------------------------------+                                                                                                                                                                                                        
                    | remote |                                                   |                                                                                                                                                                                                        
                    +--------+---------------------------------------------------+                                                                                                                                                                                                        
                    | total  |    n=50, sum=19.6592, mean=0.3932, max=3.9194     |                                                                                                                                                                                                        
                    +--------+---------------------------------------------------+                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                          
[16:13:48] INFO     [ Main ] rank 1 exited successfully, waiting for other ranks({0}) to exit                                                                                                                                                                 multiprocess_launcher.py:189
    sample_id     id  label  predict_score  predict_result                                     predict_detail
0           0  133.0      1       0.808281               1  "{'0': 0.19171901680287484, '1': 0.80828098319...
1           5  274.0      0       0.217443               0  "{'0': 0.7825566322220319, '1': 0.217443367777...
2           6  420.0      1       0.807795               1  "{'0': 0.19220495619792977, '1': 0.80779504380...
3           7   76.0      1       0.807795               1  "{'0': 0.19220495619792977, '1': 0.80779504380...
4           8  315.0      1       0.807795               1  "{'0': 0.19220495619792977, '1': 0.80779504380...
..        ...    ...    ...            ...             ...                                                ...
564       491  449.0      0       0.192732               0  "{'0': 0.8072681328604487, '1': 0.192731867139...
565       492  564.0      0       0.192732               0  "{'0': 0.8072681328604487, '1': 0.192731867139...
566       493  297.0      0       0.659294               1  "{'0': 0.3407063428915753, '1': 0.659293657108...
567       503  453.0      1       0.808056               1  "{'0': 0.19194412168367936, '1': 0.80805587831...
568       504  357.0      1       0.807795               1  "{'0': 0.19220495619792977, '1': 0.80779504380...

[569 rows x 6 columns]
[16:13:52] INFO     [Rank:0] Total: 47.6224s, Driver: 21.0738s(44.25%), Federation: 3.4240s(7.19%), Computing: 23.1245s(48.56%)                                                                                                                                            _profile.py:279
           INFO     [Rank:0]                                                                                                                                                                                                                                               _profile.py:290
                    Computing:                                                                                                                                                                                                                                                            
                    +----------+---------------------------------------------------------+                                                                                                                                                                                                
                    | function |       function          n    sum(s)   mean(s)   max(s)  |                                                                                                                                                                                                
                    |          | --------------------- ----- -------- --------- -------- |                                                                                                                                                                                                
                    |          |       mapValues        252    8.27     0.033    0.362   |                                                                                                                                                                                                
                    |          |    applyPartitions     52    5.256     0.101    4.065   |                                                                                                                                                                                                
                    |          |         join           133   4.116     0.031     0.05   |                                                                                                                                                                                                
                    |          |     mapPartitions      38    2.676     0.07      0.11   |                                                                                                                                                                                                
                    |          |  mapReducePartitions   16    2.131     0.133    0.178   |                                                                                                                                                                                                
                    |          |         first          55     0.35     0.006    0.013   |                                                                                                                                                                                                
                    |          |         take           55     0.12     0.002    0.007   |                                                                                                                                                                                                
                    |          |      parallelize        5    0.079     0.016    0.054   |                                                                                                                                                                                                
                    |          |         union           2    0.052     0.026    0.028   |                                                                                                                                                                                                
                    |          |        reduce           9    0.051     0.006    0.009   |                                                                                                                                                                                                
                    |          |   repartition_with     135   0.012      0.0      0.0    |                                                                                                                                                                                                
                    |          |         count           7    0.007     0.001    0.001   |                                                                                                                                                                                                
                    |          |        collect         70    0.005      0.0      0.0    |                                                                                                                                                                                                
                    +----------+---------------------------------------------------------+                                                                                                                                                                                                
                    |  total   |       n=829, sum=23.1245, mean=0.0279, max=4.0651       |                                                                                                                                                                                                
                    +----------+---------------------------------------------------------+                                                                                                                                                                                                
                                                                                                                                                                                                                                                                                          
                    Federation:                                                                                                                                                                                                                                                           
                    +--------+---------------------------------------------------+                                                                                                                                                                                                        
                    |  get   |       name       | n | sum(s) | mean(s) | max(s)  |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |       hist       | 9 | 1.495  |  0.166  | 0.209   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |    updated_pos   | 1 | 0.771  |  0.771  | 0.771   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |   updated_data   | 9 | 0.505  |  0.056  |  0.08   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |  new_sample_pos  | 9 |  0.31  |  0.034  | 0.038   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |       en_gh      | 3 | 0.115  |  0.038  | 0.039   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |    sync_nodes    | 9 | 0.082  |  0.009  |  0.01   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |      en_kit      | 3 | 0.059  |  0.02   | 0.022   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |  pending_samples | 1 | 0.034  |  0.034  | 0.034   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |     pack_info    | 3 | 0.025  |  0.008  | 0.009   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |     need_stop    | 2 | 0.017  |  0.008  | 0.009   |                                                                                                                                                                                                        
                    |        | -----------------+---+--------+---------+-------- |                                                                                                                                                                                                        
                    |        |     tree_dim     | 1 |  0.01  |  0.01   |  0.01   |                                                                                                                                                                                                        
                    +--------+---------------------------------------------------+                                                                                                                                                                                                        
                    | remote |                                                   |                                                                                                                                                                                                        
                    +--------+---------------------------------------------------+                                                                                                                                                                                                        
                    | total  |     n=50, sum=3.4240, mean=0.0685, max=0.7707     |                                                                                                                                                                                                        
                    +--------+---------------------------------------------------+                                                                                                                                                                                                        
                                                                                                                                                                                                                                                                                          
[16:13:52] INFO     [ Main ] rank 0 exited successfully, waiting for other ranks(set()) to exit                                                                                                                                                               multiprocess_launcher.py:189
           INFO     [ Main ] all processes exited                                                                                                                                                                                                             multiprocess_launcher.py:222
           INFO     [ Main ] cleaning up                                                                                                                                                                                                                      multiprocess_launcher.py:223
[16:14:02] INFO     [ Main ] done                                                                                                                                                                                                                             multiprocess_launcher.py:225

```
