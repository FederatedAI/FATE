import os

from setuptools import find_packages, setup

import fate

# Base requirements
install_requires = [
    "lmdb==1.3.0",
    "torch==1.13.1",
    "fate_utils",
    "pydantic==1.10.12",
    "cloudpickle==2.1.0",
    "click",
    "ruamel.yaml==0.16",
    "scikit-learn==1.2.1; sys_platform == 'darwin' and platform_machine == 'arm64'",
    "scikit-learn==1.0.1; sys_platform != 'darwin' or platform_machine != 'arm64'",
    "numpy",
    "pandas",
    "transformers",
    "accelerate",
    "beautifultable",
    "requests",
    "grpcio",
    "protobuf",
]

# Extra requirements
extras_require = {
    "rabbitmq": ["pika==1.2.1"],
    "pulsar": [
        "pulsar-client==2.10.2; sys_platform != 'darwin'",
        "pulsar-client==2.10.1; sys_platform == 'darwin'",
        "urllib3==1.26.5"
    ],
    "spark": ["pyspark"],
    "eggroll": [
        "grpcio==1.46.3",
        "grpcio-tools==1.46.3",
        "numba==0.56.4",
        "protobuf==3.19.6",
        "mmh3==3.0.0",
        "cachetools>=3.0.0",
        "cloudpickle==2.1.0",
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
    extras_require=extras_require,
    python_requires=">=3.8",
)
