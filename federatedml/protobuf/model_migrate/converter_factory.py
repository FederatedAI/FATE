from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from federatedml.protobuf.model_migrate.converter.tree_model_converter import HeteroSBTConverter
import os


def get_module_list():
    file_list = os.listdir('../../conf/setting_conf')
    module_list = [s.replace('.json', '') for s in file_list]
    return module_list


def converter_factory(module_name: str) -> ProtoConverterBase:

    module_list = get_module_list()
    if module_name not in module_list:
        raise ValueError('module name "{}" is illegal'.format(module_name))

    if module_name == 'HeteroSecureBoost':
        return HeteroSBTConverter()
    else:
        raise ValueError('this module has no converter')

