FATE
================================

FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.

https://fate.fedai.org

.. toctree::
   :caption: Deploy
   :glob:
   :maxdepth: 2

   Standalone <_build_temp/deploy/standalone-deploy/README>
   Cluster <_build_temp/deploy/cluster-deploy/README>


.. toctree::
   :caption: Quick Start
   :maxdepth: 3

   Quick Start <_build_temp/doc/tutorial/pipeline/pipeline_guide>
   Data Upload <_build_temp/doc/tutorial/dsl_conf/upload_data_guide>
   V1 Configuration <_build_temp/doc/tutorial/dsl_conf/dsl_conf_v1_setting_guide>
   V2 Configuration <_build_temp/doc/tutorial/dsl_conf/dsl_conf_v2_setting_guide>

.. toctree::
   :maxdepth: 3
   :caption: Algorithms

   FederatedML <_build_temp/doc/api/federatedml>
   Examples <_build_temp/examples/README>

.. toctree::
   :maxdepth: 3
   :caption: FATE FLOW

   FATE FLOW Guide <_build_temp/python/fate_flow/README>
   CLI API <_build_temp/python/fate_flow/doc/fate_flow_cli>
   REST API <_build_temp/python/fate_flow/doc/fate_flow_http_api>

.. toctree::
   :maxdepth: 3
   :caption: FATE Clients

   Flow SDK <_build_temp/doc/api/flow_sdk>
   Flow Client <_build_temp/doc/api/flow_client>
   Pipeline <_build_temp/doc/api/pipeline>

.. toctree::
   :maxdepth: 3
   :caption: FATE Test

   FATE TEST <_build_temp/doc/api/fate_test>


.. toctree::
   :maxdepth: 3
   :caption: Develop

   Develop Guide <_build_temp/doc/community/develop_guide>

.. toctree::
   :maxdepth: 3
   :glob:
   :caption: API

   _build_temp/doc/api/*

.. toctree::
   :maxdepth: 2
   :caption: Materials

   _build_temp/doc/resources/materials
