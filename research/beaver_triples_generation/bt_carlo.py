import time

from arch.api.utils import log_utils
from federatedml.util.transfer_variable import BeaverTripleTransferVariable
from research.beaver_triples_generation.bt_base import BaseBeaverTripleGeneration
from research.beaver_triples_generation.carlo import carlo_deal_data
from federatedml.util import consts

LOGGER = log_utils.getLogger()


class BeaverTripleGenerationCarlo(BaseBeaverTripleGeneration):

    def __init__(self, mul_ops_def, transfer_variable: BeaverTripleTransferVariable):
        self.transfer_variable = transfer_variable
        self.mul_ops_def = mul_ops_def

    def generate(self):
        LOGGER.info("@ start carlo beaver triples generation")

        start_time = time.time()

        LOGGER.info("@ get carlo beaver triple shares from a")
        party_a_bt_map_to_carlo = self._do_get(name=self.transfer_variable.party_a_bt_map_to_carlo.name,
                                               tag=self.transfer_variable.generate_transferid(
                                                   self.transfer_variable.party_a_bt_map_to_carlo),
                                               idx=-1)[0]

        LOGGER.info("@ get carlo beaver triple shares from b")
        party_b_bt_map_to_carlo = self._do_get(name=self.transfer_variable.party_b_bt_map_to_carlo.name,
                                               tag=self.transfer_variable.generate_transferid(
                                                   self.transfer_variable.party_b_bt_map_to_carlo),
                                               idx=-1)[0]

        carlo_bt_map_to_party_a, carlo_bt_map_to_party_b = carlo_deal_data(party_a_bt_map_to_carlo,
                                                                           party_b_bt_map_to_carlo,
                                                                           self.mul_ops_def)

        LOGGER.info("@ send carlo beaver triple shares to a")
        self._do_remote(carlo_bt_map_to_party_a, name=self.transfer_variable.carlo_bt_map_to_party_a.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.carlo_bt_map_to_party_a),
                        role=consts.GUEST,
                        idx=-1)

        LOGGER.info("@ send carlo beaver triple shares to b")
        self._do_remote(carlo_bt_map_to_party_b, name=self.transfer_variable.carlo_bt_map_to_party_b.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.carlo_bt_map_to_party_b),
                        role=consts.HOST,
                        idx=-1)

        end_time = time.time()

        LOGGER.info("@ carlo beaver triples generation finished!")
        LOGGER.info("@ running time: " + str(end_time - start_time))
