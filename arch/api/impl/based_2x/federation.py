import asyncio
from typing import Union

from arch.api.base.federation import Rubbish, Party, Federation
from arch.api.impl.based_2x.session import FateSession
from arch.api.impl.based_2x.table import DTable
from arch.api.utils import file_utils
from arch.api.utils import log_utils
from eggroll.core.meta_model import ErEndpoint
from eggroll.roll_pair.roll_pair import RollPair

OBJECT_STORAGE_NAME = "__federation__"
STATUS_TABLE_NAME = "__status__"

LOGGER = log_utils.getLogger()


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


# def _remote__object_key(*args):
#     return DELIM.join(["{}".format(arg) for arg in args])


class FederationRuntime(Federation):

    def __init__(self, session_id, runtime_conf):
        super().__init__(session_id, runtime_conf)
        self.rpc, self.rsc = init_roll_site_context(runtime_conf, session_id)
        self._loop = asyncio.get_event_loop()
        self.role = runtime_conf.get("local").get("role")

    def get(self, name, tag, parties: Union[Party, list]):
        if isinstance(parties, Party):
            parties = [parties]

        rubbish = Rubbish(name, tag)
        rtn = []
        rs = self.rsc.load(name=name, tag=tag)
        rs_parties = []
        for party in parties:
            rs_parties.append((party.role, party.party_id))

        # TODO:0: check if exceptions are swallowed
        futures = rs.pull(parties=rs_parties)
        for future in futures:
            obj = future.result()
            LOGGER.info(f'federation got data. tag: {tag}')
            if isinstance(obj, RollPair):
                rtn.append(DTable.from_dtable(obj.ctx.get_session().get_session_id(), obj))
                rubbish.add_table(obj)
                LOGGER.debug(f'federation got roll pair count: {obj.count()} for {tag}')
            else:
                rtn.append(obj)
        return rtn, rubbish

    def remote(self, obj, name, tag, parties):
        if isinstance(parties, Party):
            parties = [parties]

        rs_parties = []
        for party in parties:
            rs_parties.append((party.role, party.party_id))

        rubbish = Rubbish(name=name, tag=tag)
        rs = self.rsc.load(name=name, tag=tag)
        futures = rs.push(obj=obj, parties=rs_parties)

        def done_callback(fut):
            LOGGER.info("federation remote done called:{}".format(future.result()))

        for future in futures:
            future.add_done_callback(done_callback)
        return rubbish
