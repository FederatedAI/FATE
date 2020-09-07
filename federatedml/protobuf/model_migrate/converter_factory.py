from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from federatedml.protobuf.model_migrate.converter.pearson_model_converter import HeteroPearsonConverter
from federatedml.protobuf.model_migrate.converter.tree_model_converter import HeteroSBTConverter


def converter_factory(module_name: str) -> ProtoConverterBase:

    if module_name == 'HeteroSecureBoost':
        return HeteroSBTConverter()
    if module_name == "HeteroPearson":
        return HeteroPearsonConverter()
    else:
        raise ValueError('this module has no converter')

