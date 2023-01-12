import os
import sys

from setuptools import find_packages, setup

fate_path = os.path.abspath(os.path.join(__file__, os.path.pardir, "python"))
if fate_path not in sys.path:
    sys.path.append(fate_path)

import fate

packages = find_packages("python")
package_dir = {"": "python"}
install_requires = [
    "sklearn",
    "pandas",
    "protobuf",
    "pydantic",
    "click",
    "typing-extensions",
    "ruamel-yaml",
    "requests",
    "cloudpickle",
    "lmdb",
    "numpy",
    "torch",
    "rust_paillier",
    "pika",
    "pulsar",
    "urllib3",
    "grpcio",
    "ml_metadata",
    "beautifultable",
]
setup(
    name="fate",
    version=fate.__version__,
    keywords=["federated learning"],
    author="FederatedAI",
    author_email="contact@FedAI.org",
    description="FATE (Federated AI Technology Enabler) is the world's first industrial grade federated learning open source framework to enable enterprises and institutions to collaborate on data while protecting data security and privacy. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). Supporting various federated learning scenarios, FATE now provides a host of federated learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning.",
    license="Apache-2.0 License",
    url="https://fate.fedai.org/",
    packages=packages,
    package_dir=package_dir,
    install_requires=install_requires,
    python_requires=">=3.8",
)
