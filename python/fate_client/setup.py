# -*- coding: utf-8 -*-
from setuptools import setup

packages = ['flow_client',
            'flow_client.flow_cli',
            'flow_client.flow_cli.commands',
            'flow_client.flow_cli.utils',
            'flow_sdk',
            'flow_sdk.client',
            'flow_sdk.client.api',
            'pipeline',
            'pipeline.backend',
            'pipeline.component',
            'pipeline.component.nn',
            'pipeline.component.nn.backend',
            'pipeline.component.nn.backend.keras',
            'pipeline.component.nn.backend.pytorch',
            'pipeline.component.nn.backend.tf',
            'pipeline.component.nn.models',
            'pipeline.demo',
            'pipeline.interface',
            'pipeline.param',
            'pipeline.parser',
            'pipeline.test',
            'pipeline.utils',
            'pipeline.utils.invoker']

package_data = \
    {'': ['*']}

install_requires = \
    ['click>=7.1.2,<8.0.0',
     'flask>=1.0.2,<2.0.0',
     'loguru>=0.5.1,<0.6.0',
     'requests>=2.24.0,<3.0.0',
     'requests_toolbelt>=0.9.1,<0.10.0',
     'ruamel.yaml>=0.16.10,<0.17.0',
     'tensorflow==1.15.2',
     'torch==1.4.0']

entry_points = \
    {'console_scripts': ['flow = flow_client.flow:flow_cli',
                         'pipeline = pipeline.pipeline_cli:cli']}

setup_kwargs = {
    'name': 'fate-client',
    'version': '0.1.0',
    'description': 'Clients for FATE, including flow_client and pipeline',
    'long_description': 'flow_client\n===========\n\nflow_sdk\n=========\n\n\ntestsuite\n==============\n\nA useful script to running FATE\'s testsuites.\n\nquick start\n-----------\n\n1. (optional) create virtual env\n\n   .. code-block:: bash\n\n      python -m venv venv\n      source venv/bin/activate\n\n\n2. install fate_testsuite\n\n   .. code-block:: bash\n\n      pip install fate_testsuite\n\n\n3. new and edit the testsuite_config.yaml\n\n   .. code-block:: bash\n\n      # create a testsuite_config.yaml in current dir\n      testsuite config new\n      # edit priority config file with system default editor\n      # filling some field according to comments\n      testsuite config edit\n\n\n4. run some testsuites\n\n   .. code-block:: bash\n\n      testsuite suite -i <path contains *testsuite.json>\n\n5. useful logs or exception will be saved to logs dir with namespace showed in last step\n\ntestsuite_config.yaml examples\n------------------------------\n\n1. deploy multiple flow services in single node: `ip:port`, and run testsuite in different node:\n\n   - party 9999, port 9380\n   - party 10000, port 9381\n\n   filling `ssh_tunnel` and comments out `local_services`\n\n   .. code-block:: yaml\n\n      ssh_tunnel:\n          - address: ip:port\n          - services:\n              - address: 127.0.0.1:9380\n                parties: [9999]\n              - address: 127.0.0.1:9381\n                parties: [10000]\n\n\n2. deploy multiple flow services in single node: `ip:port`, and run testsuite in same node:\n\n   - party 9999, port 9380\n   - party 10000, port 9381\n\n   filling `local_services` and commnets out `ssh_tunnel`\n\n   .. code-block:: yaml\n\n      local_services: # flow services in local\n          - address: 127.0.0.1:9380\n            parties: [9999]\n          - address: 127.0.0.1:9381\n            parties: [10000]\n\n\ncommand options\n---------------\n\n1. exclude:\n\n   .. code-block:: bash\n\n      testsuite suite -i <path1 contains *testsuite.json> -e <path2 to exclude> -e <path3 to exclude> ...\n\n   will run testsuites in `path1` but not in `path2` and `path3`\n\n2. replace:\n\n   .. code-block:: bash\n\n      testsuite suite -i <path1 contains *testsuite.json> -r \'{"maxIter": 5}\'\n\n   will find all key-value pair with key "maxIter" in `data conf` or `conf` or `dsl` and replace the value with 5',
    'author': 'FederatedAI',
    'author_email': 'contact@FedAI.org',
    'maintainer': None,
    'maintainer_email': None,
    'url': 'https://fate.fedai.org/',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.6,<4.0',
}

setup(**setup_kwargs)
