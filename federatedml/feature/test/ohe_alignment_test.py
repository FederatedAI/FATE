
import unittest
import json
from federatedml.feature.OHE_with_alignment.OHE_alignment_arbiter import OHEAlignmentArbiter

class TestOHE_alignment(unittest.TestCase):

        def test_instance(self):
             ohe_alignment_arbiter = OHEAlignmentArbiter()


             guest_columns = [json.dumps({'race_black': ['0', '1'], 'race_hispanic': ['0'], 'race_asian': ['0', '1'], 'race_other': ['0', '1'], 'electivesurgery': ['0', '1']})]       
             host_columns =  [json.dumps({'race_black': ['0', '1'], 'race_hispanic': ['0', '1'], 'race_asian': ['0', '1'], 'race_other': ['0', '1'], 'electivesurgery': ['0', '1']})]

             aligned_columns = sorted(ohe_alignment_arbiter.combineAllColumnHeaders(guest_columns,host_columns)['race_hispanic'])
             self.assertTrue(len(aligned_columns) == 2)
             self.assertEqual(['0','1'], aligned_columns)

if __name__ == '__main__':
        unittest.main()