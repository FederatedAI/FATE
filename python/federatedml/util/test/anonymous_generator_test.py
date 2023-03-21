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
import random

from federatedml.util.anonymous_generator_util import Anonymous


class TeskClassifyLabelChecker(unittest.TestCase):
    def setUp(self):
        pass

    def test_extend_columns(self):
        anonymous_generator = Anonymous(role="guest", party_id=10000)
        schema = dict(header=["feature" + str(i) for i in range(100)])
        new_schema = anonymous_generator.generate_anonymous_header(schema)
        anonymous_header = new_schema["anonymous_header"]

        extend_columns1 = ["e0", "e1", "e3"]
        anonymous_header1 = anonymous_generator.extend_columns(anonymous_header, extend_columns1)

        self.assertTrue(len(anonymous_header1) == len(anonymous_header) + len(extend_columns1))
        for i in range(len(extend_columns1)):
            idx = i + len(anonymous_header)
            self.assertTrue(anonymous_header1[idx] == "_".join(["guest", "10000", "exp", str(i)]))

        extend_columns2 = ["f0", "f1", "f2", "f3"]
        anonymous_header2 = anonymous_generator.extend_columns(anonymous_header1, extend_columns2)
        self.assertTrue(len(anonymous_header2) == len(anonymous_header) + len(extend_columns1) + len(extend_columns2))
        for i in range(len(extend_columns2)):
            idx = i + len(anonymous_header) + len(extend_columns1)
            self.assertTrue(anonymous_header2[idx] == "_".join(
                ["guest", "10000", "exp", str(i + len(extend_columns1))]))

    def test_anonymous_header_generate_with_party_id(self):
        anonymous_generator = Anonymous(role="guest", party_id=10000)
        schema = dict()
        schema["header"] = ["feature" + str(i) for i in range(100)]
        new_schema = anonymous_generator.generate_anonymous_header(schema)

        anonymous_header = new_schema["anonymous_header"]
        self.assertTrue(len(anonymous_header) == 100)
        for i in range(100):
            self.assertTrue(anonymous_header[i] == "_".join(["guest", "10000", "x" + str(i)]))

    def test_anonymous_header_generate_without_party_id(self):
        schema = dict(header=["feature" + str(i) for i in range(100)])
        new_schema = Anonymous().generate_anonymous_header(schema)

        anonymous_header = new_schema["anonymous_header"]
        self.assertTrue(len(anonymous_header) == 100)
        for i in range(100):
            self.assertTrue(anonymous_header[i] == "x" + str(i))

    def test_generate_derived_header_without_extend(self):
        schema = dict(header=["feature" + str(i) for i in range(10)])
        new_schema = Anonymous(role="guest", party_id=10000).generate_anonymous_header(schema)
        anonymous_header = new_schema["anonymous_header"]

        derived_dict = {"feature5": ["feature5_f0", "feature5_f1", "feature5_f2", "feature5_f3"],
                        "feature6": ["feature6_e1", "feature6_e2", "feature6_e3"]}

        derived_anonymous_header = Anonymous().generate_derived_header(original_header=schema["header"],
                                                                       original_anonymous_header=anonymous_header,
                                                                       derived_dict=derived_dict)

        for i in range(5, 9):
            self.assertTrue(derived_anonymous_header[i] == anonymous_header[5] + "_" + str(i - 5))

        for i in range(9, 12):
            self.assertTrue(derived_anonymous_header[i] == anonymous_header[6] + "_" + str(i - 9))

    def test_generate_derived_header_with_extend(self):
        anonymous_generator = Anonymous(role="guest", party_id=10000)
        schema = dict(header=["feature" + str(i) for i in range(100)])
        new_schema = anonymous_generator.generate_anonymous_header(schema)
        anonymous_header = new_schema["anonymous_header"]

        extend_columns1 = ["e0", "e1", "e3"]
        extend_header = schema["header"] + extend_columns1
        anonymous_header1 = anonymous_generator.extend_columns(anonymous_header, extend_columns1)

        derived_dict = {"e0": ["feature5_f0", "feature5_f1", "feature5_f2", "feature5_f3"],
                        "e3": ["feature6_e1", "feature6_e2", "feature6_e3"]}

        derived_anonymous_header = anonymous_generator.generate_derived_header(
            original_header=extend_header,
            original_anonymous_header=anonymous_header1,
            derived_dict=derived_dict)

        for i in range(100, 104):
            self.assertTrue(derived_anonymous_header[i] == anonymous_header1[100] + "_" + str(i - 100))

        for i in range(105, 104):
            self.assertTrue(derived_anonymous_header[i] == anonymous_header1[102] + "_" + str(i - 105))

    def test_update_anonymous_header_with_role(self):
        schema = dict(header=["feature" + str(i) for i in range(100)])
        anonymous_header_without_role = Anonymous().generate_anonymous_header(schema)
        schema["anonymous_header"] = anonymous_header_without_role["anonymous_header"]
        schema = Anonymous.update_anonymous_header_with_role(schema, "guest", 10000)

        anonymous_header = schema["anonymous_header"]
        for i in range(100):
            self.assertTrue(anonymous_header[i] == "_".join(["guest", "10000", "x" + str(i)]))


if __name__ == '__main__':
    unittest.main()
