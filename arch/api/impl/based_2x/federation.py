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

OBJECT_STORAGE_NAME = "__federation__"
STATUS_TABLE_NAME = "__status__"

LOGGER = log_utils.getLogger()


# noinspection PyProtectedMember
def init_roll_site_context(runtime_conf, session_id):
    from eggroll.roll_site.roll_site import RollSiteContext
    from eggroll.roll_pair.roll_pair import RollPairContext
    LOGGER.info("init_roll_site_context runtime_conf: {}".format(runtime_conf))
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
    LOGGER.info("init_roll_site_context done: {}".format(rs_context.__dict__))
    return rp_context, rs_context


class FederationRuntime(Federation):

    def __init__(self, session_id, runtime_conf):
        super().__init__(session_id, runtime_conf)
        self.rpc, self.rsc = init_roll_site_context(runtime_conf, session_id)
        self._loop = asyncio.get_event_loop()
        self.role = runtime_conf.get("local").get("role")

    def get(self, name, tag, parties: Union[Party, list]):
        rs = self.rsc.load(name=name, tag=tag)
        rubbish = Rubbish(name, tag)

        if isinstance(parties, Party):
            parties = [parties]
        rs_parties = [(party.role, party.party_id) for party in parties]

        # TODO:0: check if exceptions are swallowed
        futures = rs.pull(parties=rs_parties)
        rtn = []
        for party, future in zip(rs_parties, futures):
            obj = future.result()
            LOGGER.info(f'federation got data. name: {name}, tag: {tag}')
            if isinstance(obj, RollPair):
                rtn.append(obj)
                rubbish.add_table(obj)
                if LOGGER.isEnabledFor(logging.DEBUG):
                    LOGGER.debug(f'federation got roll pair count: {obj.count()} for name: {name}, tag: {tag}')

            elif is_split_head(obj):
                num_split = obj.num_split()
                LOGGER.info(f'federation getting split data. name: {name}, tag: {tag}, num split: {num_split}')
                split_objs = []
                for k in range(num_split):
                    _split_rs = self.rsc.load(name, tag=f"{tag}.__part_{k}")
                    split_objs.append(_split_rs.pull([party])[0].result())
                obj = split_get(split_objs)
                rtn.append(obj)

            else:
                rtn.append(obj)
        return rtn, rubbish

    def remote(self, obj, name, tag, parties):
        rs = self.rsc.load(name=name, tag=tag)
        rubbish = Rubbish(name=name, tag=tag)

        if isinstance(parties, Party):
            parties = [parties]
        rs_parties = [(party.role, party.party_id) for party in parties]

        if isinstance(obj, RollPair):
            futures = rs.push(obj=obj, parties=rs_parties)
            rubbish.add_table(obj)
        else:

            futures = []
            obj, splits = maybe_split_object(obj)
            futures.extend(rs.push(obj=obj, parties=rs_parties))
            for k, v in splits:
                _split_rs = self.rsc.load(name, tag=f"{tag}.__part_{k}")
                futures.extend(_split_rs.push(obj=v, parties=rs_parties))

        def done_callback(fut):
            if LOGGER.isEnabledFor(logging.DEBUG):
                LOGGER.debug("federation remote done called:{}".format(fut.result()))

        for future in futures:
            future.add_done_callback(done_callback)
        return rubbish
