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

import deprecated

from arch.api import RuntimeInstance


# noinspection PyUnresolvedReferences,PyProtectedMember
def init(job_id: str, runtime_conf, server_conf_path="arch/conf/server_conf.json"):
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
    server_conf_path : str
      configurations of server, remain default.

    Returns
    -------
    None
      nothing returns

    Examples
    ---------
    >>> from arch.api import federation
    >>> federation.init('job_id', runtime_conf)

    """
    builder = RuntimeInstance.BUILDER
    if builder is None:
        raise EnvironmentError("session should be initialized before federation")

    RuntimeInstance.FEDERATION = builder.build_federation(federation_id=job_id,
                                                          runtime_conf=runtime_conf,
                                                          server_conf_path="arch/conf/server_conf.json")
    RuntimeInstance.TABLE_WRAPPER = builder.build_wrapper()


def all_parties():
    """
    get all parties in order

    Returns
    -------
    list
      list of parties in order
    """
    return RuntimeInstance.FEDERATION.all_parties


def local_party():
    """
    get local party

    Returns
    -------
    Party
      local party
    """
    return RuntimeInstance.FEDERATION.local_party


def roles_to_parties(roles: list) -> list:
    """
    get parties from list of roles

    Parameters
    ----------
    roles : list
      list of roles in str type

    Returns
    -------
    list
      list of parties coresponsed to given roles in order
    """ 
    return RuntimeInstance.FEDERATION.roles_to_parties(roles)


@deprecated.deprecated(version='1.2.0', reason="please use `get` api in transfer_variable instead")
def get(name, tag: str, idx=-1):
    """
    This method will block until the remote object is fetched.

    Parameters
    ----------
    name : str
      {algorithm}.{variableName} defined in transfer_conf.json.
    tag: str
      transfer version, often indicates epoch number in algorithms 
    id x: int
      idx of the party_ids in runtime role list, if idx < 0, list of all objects will be returned.

    Returns
    -------
    int or list
      The object itself if idx is in range, else return list of all objects from possible source.

    Examples
    --------
    >>> b = federation.get("RsaIntersectTransferVariable.rsa_pubkey", tag="{}".format(_tag), idx=-1)
    """
    src_role = RuntimeInstance.FEDERATION.authorized_src_roles(name)[0]
    if idx < 0:
        src_parties = RuntimeInstance.FEDERATION.roles_to_parties(roles=[src_role])
        rtn = RuntimeInstance.FEDERATION.get(name=name, tag=tag, parties=src_parties)[0]
        return [RuntimeInstance.TABLE_WRAPPER.boxed(value) for idx, value in enumerate(rtn)]
    else:
        src_node = RuntimeInstance.FEDERATION.role_to_party(role=src_role, idx=idx)
        rtn = RuntimeInstance.FEDERATION.get(name=name, tag=tag, parties=src_node)[0][0]
        return RuntimeInstance.TABLE_WRAPPER.boxed(rtn)


@deprecated.deprecated(version='1.2.0', reason="please use `remote` api in transfer_variable instead")
def remote(obj, name: str, tag: str, role=None, idx=-1):
    """
    This method will send an object to other parties.

    Parameters
    ----------
    obj : any
      The object itself which can be pickled Or a DTable
    name : str
      {algorithm}.{variableName} defined in transfer_conf.json.
    tag : str
      transfer version, often indicates epoch number in algorithms 
    role : str or None
      The role you want to send to. None indicate broadcasting.
    idx : int
      The idx of the party_ids of the role, if idx < 0, will send to all parties of the role.
    
    Returns
    -------
    None
      nothing returns

    Examples
    --------
    >>> federation.remote(a, "RsaIntersectTransferVariable.rsa_pubkey", tag="{}".format(_tag))
    """
    obj = RuntimeInstance.TABLE_WRAPPER.unboxed(obj)
    if idx >= 0 and role is None:
        raise ValueError("role cannot be None if idx specified")
    if idx >= 0:
        dst_node = RuntimeInstance.FEDERATION.role_to_party(role=role, idx=idx)
        return RuntimeInstance.FEDERATION.remote(obj=obj, name=name, tag=tag, parties=dst_node)
    else:
        if role is None:
            role = RuntimeInstance.FEDERATION.authorized_dst_roles(name)
        if isinstance(role, str):
            role = [role]
        dst_nodes = RuntimeInstance.FEDERATION.roles_to_parties(role)
        return RuntimeInstance.FEDERATION.remote(obj=obj, name=name, tag=tag, parties=dst_nodes)
