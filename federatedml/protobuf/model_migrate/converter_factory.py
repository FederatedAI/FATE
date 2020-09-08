from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from federatedml.protobuf.model_migrate.converter.tree_model_converter import HeteroSBTConverter
from federatedml.protobuf.model_migrate.converter.binning_model_converter import FeatureBinningConverter
import os


def converter_factory(module_name: str) -> ProtoConverterBase:

    if module_name == 'HeteroSecureBoost':
        return HeteroSBTConverter()
    elif module_name == 'HeteroFastSecureBoost':
        return HeteroSBTConverter()
    elif module_name == 'HeteroFeatureBinning':
        return FeatureBinningConverter()
    else:
        return None

