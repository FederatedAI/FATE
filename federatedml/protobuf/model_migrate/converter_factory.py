from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase


def converter_factory(module_name: str) -> ProtoConverterBase:

    if module_name == 'test':
        pass

    return ProtoConverterBase()