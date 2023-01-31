import os

import fate
from setuptools import find_packages, setup

packages = find_packages(".")
install_requires = [
    "scikit-learn",
    "pandas",
    "protobuf",
    "pydantic",
    "click",
    "typing-extensions",
    "ruamel.yaml",
    "requests",
    "cloudpickle",
    "lmdb",
    "numpy",
    "torch",
    "rust_paillier",
    "urllib3",
    "grpcio",
    "ml_metadata",
    "beautifultable",
]
extras_require = {
    "rabbitmq": ["pika==1.2.1"],
    "pulsar": ["pulsar-client==2.10.2"],
    "spark": ["pyspark"],
    "eggroll": [
        "grpcio==1.46.3",
        "grpcio-tools==1.46.3",
        "numba==0.56.4",
        "protobuf==3.19.6",
        "pyarrow==6.0.1",
        "mmh3==3.0.0",
        "cachetools>=3.0.0",
        "cloudpickle==2.1.0",
        "psutil>=5.7.0",
    ],
    "all": ["pyfate[rabbitmq,pulsar,spark,eggroll]"],
}

readme_path = os.path.abspath(os.path.join(__file__, os.path.pardir, os.path.pardir, "README.md"))
if os.path.exists(readme_path):
    with open(readme_path, "r") as f:
        long_description = f.read()
else:
    long_description = "fate"

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
    packages=packages,
    install_requires=install_requires,
    extras_require=extras_require,
    python_requires=">=3.8",
)
