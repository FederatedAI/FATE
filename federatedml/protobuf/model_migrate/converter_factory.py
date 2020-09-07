from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from federatedml.protobuf.model_migrate.converter.tree_model_converter import HeteroSBTConverter
import os


def converter_factory(module_name: str) -> ProtoConverterBase:

    if module_name == 'HeteroSecureBoost':
        return HeteroSBTConverter()
    elif module_name == 'HeteroFastSecureBoost':
        return HeteroSBTConverter()
    else:
        return None

