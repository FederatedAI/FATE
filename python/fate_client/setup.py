# -*- coding: utf-8 -*-
from setuptools import setup

packages = [
    "flow_client",
    "flow_client.flow_cli",
    "flow_client.flow_cli.commands",
    "flow_client.flow_cli.utils",
    "flow_sdk",
    "flow_sdk.client",
    "flow_sdk.client.api",
    "pipeline",
    "pipeline.backend",
    "pipeline.component",
    "pipeline.component.nn",
    "pipeline.component.nn.backend",
    "pipeline.component.nn.backend.keras",
    "pipeline.component.nn.backend.pytorch",
    "pipeline.component.nn.backend.tf",
    "pipeline.component.nn.models",
    "pipeline.demo",
    "pipeline.interface",
    "pipeline.param",
    "pipeline.parser",
    "pipeline.runtime",
    "pipeline.test",
    "pipeline.utils",
    "pipeline.utils.invoker",
]

package_data = {"": ["*"]}

install_requires = [
    "click>=7.1.2,<8.0.0",
    "flask>=1.0.2,<2.0.0",
    "loguru>=0.5.1,<0.6.0",
    "poetry>=0.12",
    "requests>=2.24.0,<3.0.0",
    "requests_toolbelt>=0.9.1,<0.10.0",
    "ruamel.yaml>=0.16.10,<0.17.0",
    "setuptools>=50.0,<51.0",
]

entry_points = {
    "console_scripts": [
        "flow = flow_client.flow:flow_cli",
        "pipeline = pipeline.pipeline_cli:cli",
    ]
}

setup_kwargs = {
    "name": "fate-client",
    "version": "1.8.0",
    "description": "Clients for FATE, including flow_client and pipeline",
    "long_description": "FATE Client\n===========\n\nTools for interacting with FATE.\n\nquick start\n-----------\n\n1. (optional) create virtual env\n\n   .. code-block:: bash\n\n      python -m venv venv\n      source venv/bin/activate\n\n\n2. install FATE Client\n\n   .. code-block:: bash\n\n      pip install fate-client\n\n\nPipeline\n========\n\nA high-level python API that allows user to design, start,\nand query FATE jobs in a sequential manner. For more information,\nplease refer to this `guide <./pipeline/README.rst>`__\n\nInitial Configuration\n---------------------\n\n1. Configure server information\n\n   .. code-block:: bash\n\n      # configure values in pipeline/config.yaml\n      # use real ip address to configure pipeline\n      pipeline init --ip 127.0.0.1 --port 9380 --log-directory ./logs\n\n\nFATE Flow Command Line Interface (CLI) v2\n=========================================\n\nA command line interface providing series of commands for user to design, start,\nand query FATE jobs. For more information, please refer to this `guide <./flow_client/README.rst>`__\n\nInitial Configuration\n---------------------\n\n1. Configure server information\n\n   .. code-block:: bash\n\n      # configure values in conf/service_conf.yaml\n      flow init -c /data/projects/fate/conf/service_conf.yaml\n      # use real ip address to initialize cli\n      flow init --ip 127.0.0.1 --port 9380\n\n",
    "author": "FederatedAI",
    "author_email": "contact@FedAI.org",
    "maintainer": None,
    "maintainer_email": None,
    "url": "https://fate.fedai.org/",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "entry_points": entry_points,
    "python_requires": ">=3.6,<4.0",
}


setup(**setup_kwargs)
