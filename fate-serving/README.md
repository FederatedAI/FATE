### FATE-Serving

#### Introduction

FATE-Serving is a high-performance, industrialized serving system for federated learning models, designed for production enviroments.

**It now supports**:

- Dynamic loading federated learning models
- Can serve multiple models, or multiple versions of the same model
- Support A/B testing experimental models
- Real-time inference using federated learning models
- Add multi-level cache for remote party federated inference result

#### Deploy

- Compile in the fate-serving directory
- Create your serving directory by referring to the cluster-deploy/example-dir-tree/serving-server directory
- Copy fate-serving/serving-server/target/fate-serving-server-*.jar to serving-server directory
- Copy fate-serving/serving-server/target/lib to serving-server directory
- Copy fate-serving/serving-server/src/main/resources/* to serving-server/conf
- Using the service.sh script to start/stop/restart

#### Publish Model

Please use Task Manager Client which in the arch/task_manager to operate

##### Publish Load Model

Request to load the model into serving, serving will load the model into the process cache.

```shell
python task_manager_client.py -f load -c example_conf/publish_load_model.json
```

please refer to this configuration json and set role, party, model dtable info.

##### Publish Online Model

Request to use the model as the default model for the party or the default version of the namespace model.

```shell
python task_manager_client.py -f online -c example_conf/publish_online_model.json
```

please refer to this configuration json and set role, party, model dtable info.

#### Inference

Serving currently supports three inference-related interfaces, using the grpc protocol.

- inference: Initiate an inference request and get the result
- startInferenceJob: Initiate an inference request task without getting results
- getInferenceResult: Get the result of the inference by caseid

```shell
python examples/inference_request.py ${sering_host}
```

please refer to this script for inference.

#### Adapter

Serving supports pre-processing, post-processing and data-access adapters for the actural production.

- pre-processing: Data preprocessing before model calculation
- post-processing: Data postprocessing after model calculation
- data-access: get feature from party's system

At the current stage, you need to put the java code to recompile, and later support to dynamically load the jar in the form of a release.

For now:

- push your pre-processing and post-processing adapter code into fate-serving/serving-server/src/main/java/com/webank/ai/fate/serving/adapter/processing and modify the InferencePreProcessingAdapter/InferencePostProcessingAdapter configuration parameters.
- push your data-access adapter code into fate-serving/serving-server/src/main/java/com/webank/ai/fate/serving/adapter/dataaccess and modify the OnlineDataAccessAdapter configuration parameters.

please refer to PassPostProcessing, PassPreProcessing, TestFile adapter.

#### Remote party multi-level cache

For federal learning, one inference needs to be calculated by multiple parties. In the production environment, the parties are deployed in different IDCs, and the network communication between multiple parties is one of the bottleneck.

So, fate-serving supports caches multi-party model inference results on the initiator, but never caches feature data. you can turn the remoteModelInferenceResultCacheSwitch which in the configuration.