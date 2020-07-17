import asyncio
import logging
from typing import Union

from arch.api.base.federation import Rubbish, Party, Federation
from arch.api.impl.based_2x.session import FateSession
from arch.api.utils import file_utils
from arch.api.utils import log_utils
from arch.api.utils.splitable import maybe_split_object, is_split_head, split_get
from eggroll.core.meta_model import ErEndpoint
from eggroll.roll_pair.roll_pair import RollPair
from eggroll.roll_site.roll_site import RollSite

OBJECT_STORAGE_NAME = "__federation__"
STATUS_TABLE_NAME = "__status__"

LOGGER = log_utils.getLogger()

_remote_tag_histories = set()
_get_tag_histories = set()


# noinspection PyProtectedMember
def init_roll_site_context(runtime_conf, session_id):
    from eggroll.roll_site.roll_site import RollSiteContext
    from eggroll.roll_pair.roll_pair import RollPairContext
    LOGGER.debug("init_roll_site_context runtime_conf: {}".format(runtime_conf))
    session_instance = FateSession.get_instance()._eggroll.get_session()
    rp_context = RollPairContext(session_instance)

    role = runtime_conf.get("local").get("role")
    party_id = str(runtime_conf.get("local").get("party_id"))
    _path = file_utils.get_project_base_directory() + "/arch/conf/server_conf.json"

    server_conf = file_utils.load_json_conf(_path)
    host = server_conf.get('servers').get('proxy').get("host")
    port = server_conf.get('servers').get('proxy').get("port")

    options = {'self_role': role,
               'self_party_id': party_id,
               'proxy_endpoint': ErEndpoint(host, int(port))
               }

    rs_context = RollSiteContext(session_id, rp_ctx=rp_context, options=options)
    LOGGER.debug("init_roll_site_context done: {}".format(rs_context.__dict__))
    return rp_context, rs_context


class FederationRuntime(Federation):

    def __init__(self, session_id, runtime_conf):
        super().__init__(session_id, runtime_conf)
        self.rpc, self.rsc = init_roll_site_context(runtime_conf, session_id)
        self._loop = asyncio.get_event_loop()
        self.role = runtime_conf.get("local").get("role")

    def get(self, name, tag, parties: Union[Party, list]):
        if isinstance(parties, Party):
            parties = [parties]
        self._get_side_auth(name, parties)

        rs = self.rsc.load(name=name, tag=tag)
        rubbish = Rubbish(name, tag)

        rs_parties = [(party.role, party.party_id) for party in parties]

        for party in parties:
            if (name, tag, party) in _get_tag_histories:
                raise EnvironmentError(f"get duplicate tag {(name, tag)}")
            _get_tag_histories.add((name, tag, party))

        # TODO:0: check if exceptions are swallowed
        futures = rs.pull(parties=rs_parties)
        rtn = []
        for party, future in zip(rs_parties, futures):
            obj = future.result()
            if obj is None:
                raise EnvironmentError(f"federation get None from {party} with name {name}, tag {tag}")

            LOGGER.debug(f'federation got data. name: {name}, tag: {tag}')
            if isinstance(obj, RollPair):
                rtn.append(obj)
                rubbish.add_table(obj)
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f'federation got roll pair with count {obj.count()}, name {name}, tag {tag}')

            elif is_split_head(obj):
                num_split = obj.num_split()
                LOGGER.debug(f'federation getting split data with name {name}, tag {tag}, num split {num_split}')
                split_objs = []
                for k in range(num_split):
                    _split_rs = self.rsc.load(name, tag=f"{tag}.__part_{k}")
                    split_objs.append(_split_rs.pull([party])[0].result())
                obj = split_get(split_objs)
                rtn.append(obj)

            else:
                LOGGER.debug(f'federation get obj with type {type(obj)} from {party} with name {name}, tag {tag}')
                rtn.append(obj)
        return rtn, rubbish

    def remote(self, obj, name, tag, parties):
        if isinstance(parties, Party):
            parties = [parties]
        self._remote_side_auth(name, parties)

        if obj is None:
            raise EnvironmentError(f"federation try to remote None to {parties} with name {name}, tag {tag}")

        rs = self.rsc.load(name=name, tag=tag)
        rubbish = Rubbish(name=name, tag=tag)

        rs_parties = [(party.role, party.party_id) for party in parties]

        for party in parties:
            if (name, tag, party) in _remote_tag_histories:
                raise EnvironmentError(f"remote duplicate tag {(name, tag)}")
            _remote_tag_histories.add((name, tag, party))

        if isinstance(obj, RollPair):
            futures = rs.push(obj=obj, parties=rs_parties)
            rubbish.add_table(obj)
        else:
            LOGGER.debug(f"federation remote obj with type {type(obj)} to {parties} with name {name}, tag {tag}")
            futures = []
            obj, splits = maybe_split_object(obj)
            futures.extend(rs.push(obj=obj, parties=rs_parties))
            for k, v in splits:
                _split_rs = self.rsc.load(name, tag=f"{tag}.__part_{k}")
                futures.extend(_split_rs.push(obj=v, parties=rs_parties))

        def done_callback(fut):
            try:
                result = fut.result()
            except Exception as e:
                import os
                import signal
                import traceback
                import logging
                import sys
                exc_info = sys.exc_info()
                traceback.print_exception(*exc_info)
                pid = os.getpid()
                LOGGER.exception(f"remote fail, terminating process(pid={pid})")
                logging.shutdown()
                os.kill(pid, signal.SIGTERM)
                raise e

        for future in futures:
            future.add_done_callback(done_callback)

        # warning, temporary workaround， should be remote in near released version
        if not isinstance(obj, RollPair):
            for obj_table in _get_remote_obj_store_table(rs_parties, rs):
                rubbish.add_table(obj_table)

        return rubbish


# warning, temporary workaround， should be remote in near released version
def _get_remote_obj_store_table(parties, rollsite):
    from eggroll.roll_site.utils.roll_site_utils import create_store_name
    from eggroll.core.transfer_model import ErRollSiteHeader
    tables = []
    for role_party_id in parties:
        _role = role_party_id[0]
        _party_id = str(role_party_id[1])

        _options = {}
        obj_type = 'object'
        roll_site_header = ErRollSiteHeader(
            roll_site_session_id=rollsite.roll_site_session_id,
            name=rollsite.name,
            tag=rollsite.tag,
            src_role=rollsite.local_role,
            src_party_id=rollsite.party_id,
            dst_role=_role,
            dst_party_id=_party_id,
            data_type=obj_type,
            options=_options)
        _tagged_key = create_store_name(roll_site_header)
        namespace = rollsite.roll_site_session_id
        tables.append(rollsite.ctx.rp_ctx.load(namespace, _tagged_key))

    return tables
