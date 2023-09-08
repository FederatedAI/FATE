from fate_arch.session import get_parties
from federatedml.transfer_variable.base_transfer_variable import Variable, BaseTransferVariables
from federatedml.util import consts
from federatedml.secureprotol.diffie_hellman import DiffieHellman
from federatedml.secureprotol import PaillierEncrypt
from federatedml.secureprotol.fate_paillier import PaillierPublicKey
from federatedml.secureprotol.encrypt import PadsCipher
from federatedml.util import LOGGER
from typing import Union
import hashlib


"""
Base Transfer variable
"""


class HomoTransferBase(BaseTransferVariables):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__()
        if prefix is None:
            self.prefix = f"{self.__class__.__module__}.{self.__class__.__name__}."
        else:
            self.prefix = f"{self.__class__.__module__}.{self.__class__.__name__}.{prefix}_"
        self.server = server
        self.clients = clients

    def create_client_to_server_variable(self, name):
        name = f"{self.prefix}{name}"
        return Variable.get_or_create(name, lambda: Variable(name, self.clients, self.server))

    def create_server_to_client_variable(self, name):
        name = f"{self.prefix}{name}"
        return Variable.get_or_create(name, lambda: Variable(name, self.server, self.clients))

    @staticmethod
    def get_parties(roles):
        return get_parties().roles_to_parties(roles=roles)

    @property
    def client_parties(self):
        return self.get_parties(roles=self.clients)

    @property
    def server_parties(self):
        return self.get_parties(roles=self.server)


"""
Client & Server Communication
"""


class CommunicatorTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None, disable_gc=False):
        super().__init__(server=server, clients=clients, prefix=prefix)
        if not disable_gc:
            self.client_to_server = self.create_client_to_server_variable(name="client_to_server")
            self.server_to_client = self.create_server_to_client_variable(name="server_to_client")
        else:
            self.client_to_server = self.create_client_to_server_variable(name="client_to_server").disable_auto_clean()
            self.server_to_client = self.create_server_to_client_variable(name="server_to_client").disable_auto_clean()


class ServerCommunicator(object):

    def __init__(self, prefix=None, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST)):
        self.trans_var = CommunicatorTransVar(prefix=prefix, server=server, clients=clients)
        self._client_parties = self.trans_var.client_parties

    def get_parties(self, party_idx):
        if party_idx == -1:
            return self._client_parties
        if isinstance(party_idx, list):
            return [self._client_parties[i] for i in set(party_idx)]
        if isinstance(party_idx, int):
            return self._client_parties[party_idx]
        else:
            raise ValueError('illegal party idx {}'.format(party_idx))

    def get_obj(self, suffix=tuple(), party_idx=-1):
        party = self.get_parties(party_idx)
        return self.trans_var.client_to_server.get_parties(parties=party, suffix=suffix)

    def broadcast_obj(self, obj, suffix=tuple(), party_idx=-1):
        party = self.get_parties(party_idx)
        self.trans_var.server_to_client.remote_parties(obj=obj, parties=party, suffix=suffix)


class ClientCommunicator(object):

    def __init__(self, prefix=None, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST)):
        trans_var = CommunicatorTransVar(prefix=prefix, server=server, clients=clients)
        self.trans_var = trans_var
        self._server_parties = trans_var.server_parties

    def send_obj(self, obj, suffix=tuple()):
        self.trans_var.client_to_server.remote_parties(obj=obj, parties=self._server_parties, suffix=suffix)

    def get_obj(self, suffix=tuple()):
        return self.trans_var.server_to_client.get_parties(parties=self._server_parties, suffix=suffix)


"""
Diffie Hellman Exchange
"""


class DHTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.p_power_r = self.create_client_to_server_variable(name="p_power_r")
        self.p_power_r_bc = self.create_server_to_client_variable(name="p_power_r_bc")
        self.pubkey = self.create_server_to_client_variable(name="pubkey")


class DHServer(object):

    def __init__(self, trans_var: DHTransVar = None):
        if trans_var is None:
            trans_var = DHTransVar()
        self._p_power_r = trans_var.p_power_r
        self._p_power_r_bc = trans_var.p_power_r_bc
        self._pubkey = trans_var.pubkey
        self._client_parties = trans_var.client_parties

    def key_exchange(self):
        p, g = DiffieHellman.key_pair()
        self._pubkey.remote_parties(obj=(int(p), int(g)), parties=self._client_parties)
        pubkey = dict(self._p_power_r.get_parties(parties=self._client_parties))
        self._p_power_r_bc.remote_parties(obj=pubkey, parties=self._client_parties)


class DHClient(object):

    def __init__(self, trans_var: DHTransVar = None):
        if trans_var is None:
            trans_var = DHTransVar()
        self._p_power_r = trans_var.p_power_r
        self._p_power_r_bc = trans_var.p_power_r_bc
        self._pubkey = trans_var.pubkey
        self._server_parties = trans_var.server_parties

    def key_exchange(self, uuid: str):
        p, g = self._pubkey.get_parties(parties=self._server_parties)[0]
        r = DiffieHellman.generate_secret(p)
        gr = DiffieHellman.encrypt(g, r, p)
        self._p_power_r.remote_parties(obj=(uuid, gr), parties=self._server_parties)
        cipher_texts = self._p_power_r_bc.get_parties(parties=self._server_parties)[0]
        share_secret = {uid: DiffieHellman.decrypt(gr, r, p) for uid, gr in cipher_texts.items() if uid != uuid}
        return share_secret


"""
UUID
"""


class UUIDTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.uuid = self.create_server_to_client_variable(name="uuid")


class UUIDServer(object):

    def __init__(self, trans_var: UUIDTransVar = None):
        if trans_var is None:
            trans_var = UUIDTransVar()
        self._uuid_transfer = trans_var.uuid
        self._uuid_set = set()
        self._ind = -1
        self.client_parties = trans_var.client_parties

    # noinspection PyUnusedLocal
    @staticmethod
    def generate_id(ind, *args, **kwargs):
        return hashlib.md5(f"{ind}".encode("ascii")).hexdigest()

    def _next_uuid(self):
        while True:
            self._ind += 1
            uid = self.generate_id(self._ind)
            if uid in self._uuid_set:
                continue
            self._uuid_set.add(uid)
            return uid

    def validate_uuid(self):
        for party in self.client_parties:
            uid = self._next_uuid()
            self._uuid_transfer.remote_parties(obj=uid, parties=[party])


class UUIDClient(object):

    def __init__(self, trans_var: UUIDTransVar = None):
        if trans_var is None:
            trans_var = UUIDTransVar()
        self._uuid_variable = trans_var.uuid
        self._server_parties = trans_var.server_parties

    def generate_uuid(self):
        uid = self._uuid_variable.get_parties(parties=self._server_parties)[0]
        return uid


"""
Random Padding
"""


class RandomPaddingCipherTransVar(HomoTransferBase):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.uuid_transfer_variable = UUIDTransVar(server=server, clients=clients, prefix=self.prefix)
        self.dh_transfer_variable = DHTransVar(server=server, clients=clients, prefix=self.prefix)


class RandomPaddingCipherServer(object):

    def __init__(self, trans_var: RandomPaddingCipherTransVar = None):
        if trans_var is None:
            trans_var = RandomPaddingCipherTransVar()
        self._uuid = UUIDServer(trans_var=trans_var.uuid_transfer_variable)
        self._dh = DHServer(trans_var=trans_var.dh_transfer_variable)

    def exchange_secret_keys(self):
        LOGGER.info("synchronizing uuid")
        self._uuid.validate_uuid()

        LOGGER.info("Diffie-Hellman keys exchanging")
        self._dh.key_exchange()


class RandomPaddingCipherClient(object):

    def __init__(self, trans_var: RandomPaddingCipherTransVar = None):
        if trans_var is None:
            trans_var = RandomPaddingCipherTransVar()
        self._uuid = UUIDClient(trans_var=trans_var.uuid_transfer_variable)
        self._dh = DHClient(trans_var=trans_var.dh_transfer_variable)
        self._cipher = None

    def create_cipher(self) -> PadsCipher:
        LOGGER.info("synchronizing uuid")
        uuid = self._uuid.generate_uuid()
        LOGGER.info(f"got local uuid")

        LOGGER.info("Diffie-Hellman keys exchanging")
        exchanged_keys = self._dh.key_exchange(uuid)
        LOGGER.info(f"got Diffie-Hellman exchanged keys")

        cipher = PadsCipher()
        cipher.set_self_uuid(uuid)
        cipher.set_exchanged_keys(exchanged_keys)
        self._cipher = cipher
        return cipher

    def encrypt(self, transfer_weights):
        return self._cipher.encrypt(transfer_weights)
