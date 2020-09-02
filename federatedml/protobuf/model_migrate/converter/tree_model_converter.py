from federatedml.protobuf.model_migrate.converter.converter_base import ProtoConverterBase
from typing import List

class HeteroSBTConverter(ProtoConverterBase):

    def convert(self, model_contents: dict,
                old_guest_list: List[int],
                new_guest_list: List[int],
                old_host_list: List[int],
                new_host_list: List[int],
                old_arbiter_list: List[int] = None,
                new_arbiter_list: List[int] = None,
                ):
        pass