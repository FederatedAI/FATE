Example Usage Guide.
====================

We provide here examples of FATE jobs, including FATE-Pipeline scripts,
DSL conf files, and modeling quality comparison tasks

We suggest that user use example-runner tool `FATE-Test <../python/fate_test/README.rst>`__.

Also, for smoother interaction with FATE-Flow, we suggest that user install Flow-Client with `FATE-Client <../python/fate_client/README.rst>`__.

To quickly start model training and predictions using dsl & pipeline, please refer to：

1. `DSL v1 train and predict quick tutorial <./experiment_template/user_usage/dsl_v1_predict_tutorial.md>`__.
2. `DSL v2 train and predict quick tutorial <./experiment_template/user_usage/dsl_v2_predict_tutorial.md>`__.
3. `Pipeline train and predict quick tutorial <./experiment_template/user_usage/pipeline_predict_tutorial.md>`__.

Below lists included types of examples.

FATE-Pipeline
-------------

To enhance usability of FATE, starting at FATE-v1.5, FATE provides python APIs.
User may develop federated learning models conveniently with
`FATE-Pipeline <../python/fate_client/pipeline/README.rst>`__.
We provide a host of Pipeline examples for each FATE module and a quick start guide for Pipeline
`here <./pipeline>`__

Below shows how to build and fit a Hetero SecureBoost model with FATE-Pipeline in few lines.

.. code-block:: python

    import json
    from pipeline.backend.config import Backend, WorkMode
    from pipeline.backend.pipeline import PipeLine
    from pipeline.component import Reader, DataIO, Intersection, HeteroSecureBoost, Evaluation
    from pipeline.interface import Data
    from pipeline.runtime.entity import JobParameters

    guest_train_data = {"name": "breast_hetero_guest", "namespace": "experiment"}
    host_train_data = {"name": "breast_hetero_host", "namespace": "experiment"}

    # initialize pipeline
    pipeline = PipeLine().set_initiator(role="guest", party_id=9999).set_roles(guest=9999, host=10000)

    # define components
    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=9999).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=10000).component_param(table=host_train_data)
    dataio_0 = DataIO(name="dataio_0", with_label=True)
    dataio_0.get_party_instance(role="host", party_id=10000).component_param(with_label=False)
    intersect_0 = Intersection(name="intersection_0")
    hetero_secureboost_0 = HeteroSecureBoost(name="hetero_secureboost_0",
                                             num_trees=5,
                                             bin_num=16,
                                             task_type="classification",
                                             objective_param={"objective": "cross_entropy"},
                                             encrypt_param={"method": "iterativeAffine"},
                                             tree_param={"max_depth": 3})
    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    # add components to pipeline, in order of task execution
    pipeline.add_component(reader_0)\
        .add_component(dataio_0, data=Data(data=reader_0.output.data))\
        .add_component(intersect_0, data=Data(data=dataio_0.output.data))\
        .add_component(hetero_secureboost_0, data=Data(train_data=intersect_0.output.data))\
        .add_component(evaluation_0, data=Data(data=hetero_secureboost_0.output.data))

    # compile & fit pipeline
    pipeline.compile().fit(JobParameters(backend=Backend.EGGROLL, work_mode=WorkMode.STANDALONE))

    # query component summary
    print(f"Evaluation summary:\n{json.dumps(pipeline.get_component('evaluation_0').get_summary(), indent=4)}")

    # Evaluation summary:
    # {
    #   "auc": 0.9971790603033666,
    #   "ks": 0.9624094920987263
    # }

Code for the above job can also be found `here <./pipeline/demo/pipeline-quick-demo.py>`__.

DSL
---

DSL is the first method FATE provides for constructing federated
modeling jobs. For more information, please refer this
`guide <../doc/dsl_conf_v2_setting_guide.rst>`__ on DSL.

Upgraded DSL(DSL v2) by FATE-v1.5 comes with the following major features:

1. Predictions DSL may now be configured through FATE-Flow cli. Please note
   that with new DSL training job will no longer automatically form prediction DSL;
   user needs to first form DSL manually with FATE-Flow cli before running
   prediction task.
2. New components may now be added to prediction DSL;
   for instance, ``evaluation`` module may be added to prediction task.
3. Standardize style of ``role_parameter`` and ``algorithm_parameter``.

For DSL v2 examples, please refer `dsl/v2 <./dsl/v2>`__. For examples of
the older version, please refer `dsl/v1 <./dsl/v1>`__. This is the "federatedml-1.x-examples" in older version. Please note that
starting at version 1.6, FATE may no longer support DSL v1.

Cross Validation Task
~~~~~~~~~~~~~~~~~~~~~~~

Starting at version 1.6, cross validation tasks can output fold history data when ``output_fold_history`` is set to True.
Output data contains either prediction ``score`` or original ``instance`` value. Please note that the ``score`` output from cross validation
tasks may not be input to Evaluation module. All testsuites of modeling modules include demos
on setting `cv parameters <../python/federatedml/param/cross_validation_param.py>`_.


Benchmark Quality
-----------------

Starting at FATE-v1.5, FATE provides modeling quality verifier for comparing modeling
quality of centralized training and FATE federated modeling.
As of v1.5, we have provided quality comparison scripts for the
following common models:

* heterogeneous scenario: LogisticRegression(`benchmark_quality/hetero_lr <./benchmark_quality/hetero_lr>`__),
  SecureBoost(`benchmark_quality/hetero_sbt <./benchmark_quality/hetero_sbt>`__),
  FastSecureBoost(`benchmark_quality/hetero_fast_sbt <./benchmark_quality/hetero_fast_sbt>`__),
  NN(`benchmark_quality/hetero_nn <./benchmark_quality/hetero_nn>`__).
* homogeneous scenario: LogisticRegression(`benchmark_quality/homo_lr <./benchmark_quality/homo_lr>`__),
  SecureBoost(`benchmark_quality/homo_sbt <./benchmark_quality/homo_sbt>`__), NN(`benchmark_quality/homo_nn <./benchmark_quality/homo_nn>`__.

Starting at v1.6, benchmark quality supports matching metrics from the same script. For more details, please refer
to the `guide <../python/fate_test/README.rst>`__.

Upload Default Data
-------------------

FATE provides a collection of publicly available data at `data <./data>`__ directory,
along with a utility script for uploading all data sets. User may use the provided
script to upload all pre-given data, or modify the corresponding configuration file for uploading
arbitrary data. Please refer `scripts <./scripts/README.rst>`__ for details.

Alternatively, user may use `FATE-Test <../python/fate_test/README.rst#data>`__ for uploading data.

Toy Example
-----------

FATE provides simple toy job for quick experiment when user developing FATE modules
or testing for deployment. For details, please refer `toy_example <./toy_example/README.md>`__.


Min-test
--------

Min-test is used for deployment testing and quick modeling demo. Min-test includes
tasks of hetero Logistic Regression and hetero SecureBoost.
User only needs to configure few simple parameters to run a full modeling job
with FATE. Please refer `min_test_task <./min_test_task/README.rst>`__ for instructions.
