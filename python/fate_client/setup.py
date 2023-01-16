#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# -*- coding: utf-8 -*-
from setuptools import setup

packages = [
    "fate_client",
    "fate_client.pipeline",
    "fate_client.pipeline.component_define",
    "fate_client.pipeline.component_define.fate",
    "fate_client.pipeline.components",
    "fate_client.pipeline.components.fate",
    "fate_client.pipeline.conf",
    "fate_client.pipeline.entity",
    "fate_client.pipeline.executor",
    "fate_client.pipeline.interface",
    "fate_client.pipeline.manager",
    "fate_client.pipeline.scheduler",
    "fate_client.pipeline.utils",
    "fate_client.pipeline.utils.fateflow",
    "fate_client.pipeline.utils.standalone",
]

package_data = {"": ["*"]}

install_requires = [
    "click>=7.1.2,<8.0.0",
    "poetry>=0.12",
    "pandas>=1.1.5",
    "requests>=2.24.0,<3.0.0",
    "requests_toolbelt>=0.9.1,<0.10.0",
    "ruamel.yaml>=0.16.10",
    "setuptools>=65.5.1",
    "networkx>=2.8.7",
    "pydantic",
    "ml_metadata",
]

entry_points = {"console_scripts": ["fate_client = fate_client.cli:cli"]}

setup_kwargs = {
    "name": "fate-client",
    "version": "2.0.0-alpha",
    "description": "Clients for FATE, including fate_client and pipeline",
    "long_description_content_type": "text/markdown",
    "long_description": "Clients for FATE, including fate_client and pipeline",
    "author": "FederatedAI",
    "author_email": "contact@FedAI.org",
    "maintainer": None,
    "maintainer_email": None,
    "url": "https://fate.fedai.org/",
    "packages": packages,
    "package_data": package_data,
    "install_requires": install_requires,
    "entry_points": entry_points,
    "python_requires": ">=3.8",
}


setup(**setup_kwargs)
