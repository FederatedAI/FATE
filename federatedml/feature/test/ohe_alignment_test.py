import json
import unittest
import uuid

from arch.api import session
from federatedml.feature.ohe_with_alignment.OHE_alignment_arbiter import OHEAlignmentArbiter


class TestOHE_alignment(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)

    def test_instance(self):
        ohe_alignment_arbiter = OHEAlignmentArbiter()

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
        try:
            session.cleanup("*", self.job_id, True)
        except EnvironmentError:
            pass
        try:
            session.cleanup("*", self.job_id, False)
        except EnvironmentError:
            pass


if __name__ == '__main__':
    unittest.main()
