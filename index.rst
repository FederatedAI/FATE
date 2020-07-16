Welcome to FATE's documentation!
================================

FATE (Federated AI Technology Enabler) is an open-source project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.

https://fate.fedai.org

.. toctree::
   :caption: Install
   :maxdepth: 2

   Standalone <standalone-deploy/README>
   Cluster <cluster-deploy/README>


.. toctree::
   :caption: Quick Start
   :maxdepth: 3

   Quick Start <examples/federatedml-1.x-examples/README>
   Data Upload <doc/upload_data_guide>
   Configuration <doc/dsl_conf_setting_guide>

.. toctree::
   :maxdepth: 3
   :caption: Algorithms

   FederatedML <federatedml/README>

.. toctree::
   :maxdepth: 3
   :caption: FATE FLOW

   FATE FLOW Guide <fate_flow/README>
   CLI API <fate_flow/doc/fate_flow_cli>
   REST API <fate_flow/doc/fate_flow_rest_api>

.. toctree::
   :maxdepth: 3
   :caption: Develop

   Develop Guide <doc/develop_guide>

.. toctree::
   :maxdepth: 3
   :glob:
   :caption: API

   doc/api/*

.. toctree::
   :maxdepth: 2
   :caption: Materials

   doc/materials