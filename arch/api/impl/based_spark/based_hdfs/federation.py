
from typing import Union

from arch.api.base.federation import Federation
from arch.api.utils import file_utils
from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()



class FederationRuntime(Federation):

    def __init__(self, session_id, runtime_conf):
        super().__init__(session_id, runtime_conf)
        self.role = runtime_conf.get("local").get("role")

    def get(self, name, tag, parties: Union[Party, list]):
        pass

    def remote(self, obj, name, tag, parties):
        pass
