import time

from arch.api.utils import log_utils
from federatedml.util import consts
from federatedml.util.transfer_variable import BeaverTripleTransferVariable
from research.beaver_triples_generation.beaver_triple import PartyABeaverTripleGenerationHelper
from research.beaver_triples_generation.bt_base import BaseBeaverTripleGeneration

LOGGER = log_utils.getLogger()


class BeaverTripleGenerationGuest(BaseBeaverTripleGeneration):

    def __init__(self, party_a_bt_gene_helper: PartyABeaverTripleGenerationHelper,
                 transfer_variable: BeaverTripleTransferVariable):
        self.party_a_bt_gene_helper = party_a_bt_gene_helper
        self.transfer_variable = transfer_variable

    def generate(self):
        LOGGER.info("@ start guest beaver triples generation")

        start_time = time.time()
        self.party_a_bt_gene_helper.initialize_beaver_triples()

        party_a_bt_map_to_carlo = self.party_a_bt_gene_helper.get_to_carlo_beaver_triple_map()
        party_a_bt_map_to_b = self.party_a_bt_gene_helper.get_to_other_party_beaver_triple_map()

        self._do_remote(party_a_bt_map_to_b, name=self.transfer_variable.party_a_bt_map_to_b.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.party_a_bt_map_to_b),
                        role=consts.HOST,
                        idx=-1)

        party_b_bt_map_to_a = self._do_get(name=self.transfer_variable.party_b_bt_map_to_a.name,
                                           tag=self.transfer_variable.generate_transferid(
                                               self.transfer_variable.party_b_bt_map_to_a),
                                           idx=-1)[0]

        self._do_remote(party_a_bt_map_to_carlo, name=self.transfer_variable.party_a_bt_map_to_carlo.name,
                        tag=self.transfer_variable.generate_transferid(self.transfer_variable.party_a_bt_map_to_carlo),
                        role=consts.ARBITER,
                        idx=-1)

        carlo_bt_map_to_party_a = self._do_get(name=self.transfer_variable.carlo_bt_map_to_party_a.name,
                                               tag=self.transfer_variable.generate_transferid(
                                                   self.transfer_variable.carlo_bt_map_to_party_a),
                                               idx=-1)[0]

        self.party_a_bt_gene_helper.complete_beaver_triples(party_b_bt_map_to_a, carlo_bt_map_to_party_a)
        party_a_bt_map = self.party_a_bt_gene_helper.get_beaver_triple_map()

        end_time = time.time()
        LOGGER.info("@ running time: " + str(end_time - start_time))

        self.save_beaver_triples(party_a_bt_map, bt_map_name="guest_bt_map")
