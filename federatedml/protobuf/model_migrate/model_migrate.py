from typing import List
from federatedml.protobuf.model_migrate.converter_factory import converter_factory
import copy


def check_party_ids(new_id_list, old_id_list):

    for id0, id1 in zip(new_id_list, old_id_list):
        if type(id0) != int or type(id1) != int:
            raise ValueError('party id must be an integer')


def model_migration(model_contents: dict,
                    module_name,
                    old_guest_list: List[int],
                    new_guest_list: List[int],
                    old_host_list: List[int],
                    new_host_list: List[int],
                    old_arbiter_list=None,
                    new_arbiter_list=None,
                    ):

    # check
    check_party_ids(old_guest_list, new_guest_list)
    check_party_ids(old_host_list, new_host_list)
    if old_arbiter_list is not None and new_arbiter_list is not None:
        check_party_ids(old_arbiter_list, new_arbiter_list)
    else:
        if not (old_arbiter_list is None and new_arbiter_list is None):
            raise ValueError('arbiter lists should be all lists or all None')

    converter = converter_factory(module_name)
    if converter is None:
        raise ValueError('module {} is not found in converter list'.format(module_name))

    # replace old id with new id using converter
    model_contents_cpy = copy.deepcopy(model_contents)
    result = converter.convert(model_contents_cpy, old_guest_list, new_guest_list, old_host_list, new_host_list,
                               old_arbiter_list, new_arbiter_list)

    return result
