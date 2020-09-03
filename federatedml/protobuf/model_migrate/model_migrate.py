from typing import List
from federatedml.protobuf.model_migrate.converter_factory import converter_factory
import copy


def generate_id_mapping(old_id, new_id):

    if old_id is None and new_id is None:
        return {}
    elif not (type(old_id) == list and type(new_id) == list):
        raise ValueError('illegal input format: id lists type should be list, however got: \n'
                         'content: {}/ type: {} \n'
                         'content: {}/ type: {}'.format(old_id, type(old_id), new_id, type(new_id)))

    if len(old_id) != len(new_id):
        raise ValueError('id lists length does not match: len({}) != len({})'.format(old_id, new_id))

    mapping = {}
    for id0, id1 in zip(old_id, new_id):
        if type(id0) != int or type(id1) != int:
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

    # replace old id with new id using converter
    guest_mapping_dict = generate_id_mapping(old_guest_list, new_guest_list)
    host_mapping_dict = generate_id_mapping(old_host_list, new_host_list)
    arbiter_mapping_dict = generate_id_mapping(old_arbiter_list, new_arbiter_list)

    model_contents_cpy = copy.deepcopy(model_contents)
    result = converter.convert(model_contents_cpy['param'], model_contents_cpy['meta'], guest_mapping_dict,
                               host_mapping_dict, arbiter_mapping_dict)

    return result
