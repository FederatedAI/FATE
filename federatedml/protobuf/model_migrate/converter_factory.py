import typing

from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from federatedml.protobuf.model_migrate.converter.pearson_model_converter import HeteroPearsonConverter
from federatedml.protobuf.model_migrate.converter.tree_model_converter import HeteroSBTConverter
from federatedml.protobuf.model_migrate.converter.binning_model_converter import FeatureBinningConverter


def converter_factory(module_name: str) -> typing.Optional[ProtoConverterBase]:
    if module_name == 'HeteroSecureBoost':
        return HeteroSBTConverter()
    elif module_name == 'HeteroFastSecureBoost':
        return HeteroSBTConverter()
    elif module_name == 'HeteroPearson':
        return HeteroPearsonConverter()
    elif module_name == 'HeteroFeatureBinning':
        return FeatureBinningConverter()
    else:
        return None
