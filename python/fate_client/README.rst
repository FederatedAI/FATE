FATE Client
===========

Tools for interacting with FATE.

quick start
-----------

1. (optional) create virtual env

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate


2. install FATE Client

   .. code-block:: bash

      pip install fate-client


Pipeline
========

A high-level python API that allows user to design, start,
and query FATE jobs in a sequential manner. For more information,
please refer to this `guide <./pipeline/README.rst>`__

Initial Configuration
---------------------

1. Configure server information

   .. code-block:: bash

      # configure by conf file
      pipeline init -c pipeline/config.yaml
      # alternatively, input real ip address and port info to initialize pipeline
      # optionally, set log directory for Pipeline
      pipeline init --ip 127.0.0.1 --port 9380 --log-directory ./logs


FATE Flow Command Line Interface (CLI) v2
=========================================

A command line interface providing series of commands for user to design, start,
and query FATE jobs. For more information, please refer to this `guide <./flow_client/README.rst>`__

Initial Configuration
---------------------

1. Configure server information

   .. code-block:: bash

      # configure values in conf/service_conf.yaml
      flow init -c /data/projects/fate/conf/service_conf.yaml
      # alternatively, input real ip address and port info to initialize cli
      flow init --ip 127.0.0.1 --port 9380

