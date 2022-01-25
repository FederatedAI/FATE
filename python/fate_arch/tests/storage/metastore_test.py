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
import unittest
from fate_arch.metastore import base_model


class TestBaseModel(unittest.TestCase):
    def test_auto_date_timestamp_field(self):
        self.assertEqual(
            base_model.auto_date_timestamp_field(), {
                'write_access_time', 'create_time', 'read_access_time', 'end_time', 'update_time', 'start_time'})

    def test(self):
        from peewee import IntegerField, FloatField, AutoField, BigAutoField, BigIntegerField, BitField
        from peewee import CharField, TextField, BooleanField, BigBitField
        from fate_arch.metastore.base_model import JSONField, LongTextField
        for f in {IntegerField, FloatField, AutoField, BigAutoField, BigIntegerField, BitField}:
            self.assertEqual(base_model.is_continuous_field(f), True)
        for f in {CharField, TextField, BooleanField, BigBitField}:
            self.assertEqual(base_model.is_continuous_field(f), False)
        for f in {JSONField, LongTextField}:
            self.assertEqual(base_model.is_continuous_field(f), False)


if __name__ == '__main__':
    unittest.main()
