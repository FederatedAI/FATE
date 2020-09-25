import unittest
import uuid

from fate_arch.session import computing_session as session
from federatedml.feature.homo_onehot.homo_ohe_arbiter import HomoOneHotArbiter


class TestOHE_alignment(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)

    def test_instance(self):
        ohe_alignment_arbiter = HomoOneHotArbiter()

        guest_columns = [
            {'race_black': ['0', '1'], 'race_hispanic': ['0'], 'race_asian': ['0', '1'], 'race_other': ['1'],
             'electivesurgery': ['0', '1']}]
        host_columns = [
            {'race_black': ['0', '1'], 'race_hispanic': ['0', '1'], 'race_asian': ['0', '1'], 'race_other': ['0'],
             'electivesurgery': ['0', '1']}]

        aligned_columns = sorted(
            ohe_alignment_arbiter.combine_all_column_headers(guest_columns, host_columns)['race_hispanic'])
        self.assertTrue(len(aligned_columns) == 2)
        self.assertEqual(['0', '1'], aligned_columns)

    def tearDown(self):
        session.stop()


if __name__ == '__main__':
    unittest.main()
