from typing import List


def check_party_ids(new_id_list, old_id_list):

    for id0, id1 in zip(new_id_list, old_id_list):
        if type(id0) != int or type(id1) != int:
            raise ValueError('id must be an integer')


def model_migration(model_contents: dict,
                    old_guest_list: List[int],
                    new_guest_list: List[int],
                    old_host_list: List[int],
                    new_host_list: List[int],
                    module_name,
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

    # replace old id with new id using adapter

    # outputs new model_content
