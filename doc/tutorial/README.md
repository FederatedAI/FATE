# Usage

Here we provide tutorials on running FATE jobs:

Submitting jobs with `Pipeline` is recommended, here are some `Jupyter Notebook` with guided instructions:

- [Upload Data with FATE-Pipeline](pipeline/pipeline_tutorial_upload.ipynb)
- [Train & Predict Hetero SecureBoost with FATE-Pipeline](pipeline/pipeline_tutorial_hetero_sbt.ipynb)
- [Build NN models with FATE-Pipeline](pipeline/nn_tutorial/README.md)
- [Upload & Train Hetero SecureBoost on Data with Match ID](pipeline/pipeline_tutorial_match_id.ipynb)
- [Upload & Train Hetero SecureBoost on Data with Meta](pipeline/pipeline_tutorial_uploading_data_with_meta.ipynb)
- [Upload & Run An Intersection Task on Data with Multiple Match IDs](pipeline/pipeline_tutorial_multiple_id_columns.ipynb)

Submitting jobs without `Pipeline` is supported as well, which one should provide job configuration(s) in json format:

- [Tutorial](dsl_conf/dsl_conf_tutorial.md)
- [Upload data](dsl_conf/upload_data_guide.md)
- [DSL conf setting](dsl_conf/dsl_conf_v2_setting_guide.md)

Models can be published with FATE Serving to `Serving without FATE`:

- [publish model](model_publish_with_serving_guide.md)

And for those who want to run jobs in batches, ie. run algorithm tests, try using `fate_test`:
    
- [FATE-Test Tutorial](fate_test_tutorial.md)

To merge models from different roles and export as sklearn/LightGBM or PMML format, please refer to the tutorial below:

- [Guide to Merging FATE Model](./model_merge.md)
