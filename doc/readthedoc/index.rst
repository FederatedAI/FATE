FATE
================================

FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.

https://fate.fedai.org

.. toctree::
   :caption: Install
   :maxdepth: 2

   Standalone <_build_temp/standalone-deploy/README>
   Cluster <_build_temp/cluster-deploy/README>


.. toctree::
   :caption: Quick Start
   :maxdepth: 3

   Quick Start <_build_temp/examples/dsl/v2/README>
   Data Upload <_build_temp/doc/upload_data_guide>
   Configuration <_build_temp/doc/dsl_conf_setting_guide>

.. toctree::
   :maxdepth: 3
   :caption: Algorithms

   FederatedML <_build_temp/python/federatedml/README>

.. toctree::
   :maxdepth: 3
   :caption: FATE FLOW

   FATE FLOW Guide <_build_temp/python/fate_flow/README>
   CLI API <_build_temp/python/fate_flow/doc/fate_flow_cli>
   REST API <_build_temp/python/fate_flow/doc/fate_flow_http_api>

.. toctree::
   :maxdepth: 3
   :caption: FATE Clients

   Flow SDK <_build_temp/python/fate_client/flow_sdk/README>
   Flow Client <_build_temp/python/fate_client/flow_client/README>
   Pipeline <_build_temp/python/fate_client/pipeline/README>

.. toctree::
   :maxdepth: 3
   :caption: FATE Test

   FATE TEST <_build_temp/python/fate_test/README>


.. toctree::
   :maxdepth: 3
   :caption: Develop

   Develop Guide <_build_temp/doc/develop_guide>

.. toctree::
   :maxdepth: 3
   :glob:
   :caption: API

   _build_temp/doc/api/*

.. toctree::
   :maxdepth: 2
   :caption: Materials

   _build_temp/doc/materials
