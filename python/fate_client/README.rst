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

      # configure values in pipeline/config.yaml
      pipeline config --ip 172.0.0.1 --port 9380 --log-directory ./logs
