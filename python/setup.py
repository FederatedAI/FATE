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

import os

from setuptools import find_packages, setup

import fate

# Base requirements
install_requires = [
    "lmdb==1.4.1",
    "torch==2.3.1",
    "fate_utils",
    "pydantic==1.10.12",
    "cloudpickle==2.1.0",
    "click",
    "ruamel.yaml==0.16",
    "scikit-learn==1.4.1.post1",
    "numpy<2.0.0",
    "pandas",
    "transformers==4.37.2",
    "accelerate==0.27.2",
    "beautifultable",
    "requests<2.26.0",
    "grpcio==1.62.1",
    "protobuf==4.24.4",
    "rich==13.7.1",
    "omegaconf",
    "opentelemetry-api",
    "opentelemetry-sdk",
    "mmh3==3.0.0",
    "safetensors==0.4.1",
]

# Extra requirements
extras_require = {
    "rabbitmq": ["pika"],
    "pulsar": [
        "pulsar-client==3.4.0",
    ],
    "spark": ["pyspark"],
    "eggroll": [
        "grpcio-tools==1.62.1",
        "psutil>=5.7.0",
    ],
    "all": ["pyfate[rabbitmq,pulsar,spark,eggroll]"],
}

# Long description from README.md
readme_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, "README.md"))
if os.path.exists(readme_path):
    with open(readme_path, "r") as f:
        long_description = f.read()
else:
    long_description = "fate"

# Setup function
setup(
    name="pyfate",
    version=fate.__version__,
    keywords=["federated learning"],
    author="FederatedAI",
    author_email="contact@FedAI.org",
    long_description_content_type="text/markdown",
    long_description=long_description,
    license="Apache-2.0 License",
    url="https://fate.fedai.org/",
    packages=find_packages("."),
    install_requires=install_requires,
    include_package_data=True,
    package_data={"": ["*.yaml"]},
    extras_require=extras_require,
    python_requires=">=3.8",
)
