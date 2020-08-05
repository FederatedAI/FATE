#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from fate_arch import session
from fate_arch.federation import FederationType


def init(job_id: str, runtime_conf, *args, **kwargs):
    """
    Initializes federation module. This method should be called before calling other federation APIs

    Parameters
    ----------
    job_id : str
      job id and default table namespace of this runtime. None is ok, uuid will be used.
    runtime_conf : dict
      specifiy the role and parties. runtime_conf should be a dict with
        
        1. key "local" maps to the current process' role and party_id.
        
        2. key "role" maps to a dict mapping from each role to all involving party_ids.
        
        .. code-block:: json

          {
            "local": {
                "role": "host",
                "party_id": 1000
            }
            "role": {
                "host": [999, 1000, 1001],
                "guest": [10002]
            }
          }

    Returns
    -------
    None
      nothing returns

    Examples
    ---------
    >>> from arch.api import federation
    >>> federation.init('job_id', runtime_conf)

    """
    session.default().init_federation(federation_type=FederationType.EGGROLL,
                                      federation_session_id=job_id,
                                      runtime_conf=runtime_conf)
