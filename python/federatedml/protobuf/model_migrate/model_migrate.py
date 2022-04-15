from typing import List
from federatedml.protobuf.model_migrate.converter_factory import converter_factory
from federatedml.model_base import serialize_models
import copy


def generate_id_mapping(old_id, new_id):

    if old_id is None and new_id is None:
        return {}
    elif not (isinstance(old_id, list) and isinstance(new_id, list)):
        raise ValueError('illegal input format: id lists type should be list, however got: \n'
                         'content: {}/ type: {} \n'
                         'content: {}/ type: {}'.format(old_id, type(old_id), new_id, type(new_id)))

    if len(old_id) != len(new_id):
        raise ValueError('id lists length does not match: len({}) != len({})'.format(old_id, new_id))

    mapping = {}
    for id0, id1 in zip(old_id, new_id):
        if not isinstance(id0, int) or not isinstance(id1, int):
            raise ValueError('party id must be an integer, got {}:{} and {}:{}'.format(id0, type(id0),
                                                                                       id1, type(id1)))
        mapping[id0] = id1

    return mapping


def model_migration(model_contents: dict,
                    module_name,
                    old_guest_list: List[int],
                    new_guest_list: List[int],
                    old_host_list: List[int],
                    new_host_list: List[int],
                    old_arbiter_list=None,
                    new_arbiter_list=None,
                    ):

    converter = converter_factory(module_name)
    if converter is None:
        # no supported converter, return
        return serialize_models(model_contents)

    # replace old id with new id using converter
    guest_mapping_dict = generate_id_mapping(old_guest_list, new_guest_list)
    host_mapping_dict = generate_id_mapping(old_host_list, new_host_list)
    arbiter_mapping_dict = generate_id_mapping(old_arbiter_list, new_arbiter_list)

    model_contents_cpy = copy.deepcopy(model_contents)
    keys = model_contents.keys()
    param, meta = None, None
    param_key, meta_key = None, None
    for key in keys:
        if 'Param' in key:
            param_key = key
            param = model_contents_cpy[key]
        if 'Meta' in key:
            meta_key = key
            meta = model_contents_cpy[key]

    if param is None or meta is None:
        raise ValueError('param or meta is None')

    converted_param, converted_meta = converter.convert(param, meta, guest_mapping_dict,
                                                        host_mapping_dict, arbiter_mapping_dict)

    return serialize_models({param_key: converted_param, meta_key: converted_meta})
