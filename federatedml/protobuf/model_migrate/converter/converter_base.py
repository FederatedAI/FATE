from abc import ABC, abstractmethod
from typing import List


class ProtoConverterBase(ABC):

    @abstractmethod
    def convert(self, model_contents: dict,
                old_guest_list: List[int],
                new_guest_list: List[int],
                old_host_list: List[int],
                new_host_list: List[int],
                old_arbiter_list: List[int] = None,
                new_arbiter_list: List[int] = None,
                ):
        raise NotImplementedError('this interface is not implemented')