#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import uuid

from fate_arch.session import computing_session as session
from pipeline.backend.config import Backend
from pipeline.backend.config import WorkMode
from pipeline.backend.pipeline import PipeLine


class TestUpload(unittest.TestCase):
    def setUp(self):
        self.job_id = str(uuid.uuid1())
        session.init(self.job_id)
        self.file = "examples/data/breast_homo_guest.csv"
        self.table_name = "breast_homo_guest"
        self.data_count = 227

    def test_upload(self):
        upload_pipeline = PipeLine()
        upload_pipeline.add_upload_data(file=self.file,
                                        table_name=self.table_name, namespace=self.job_id)
        upload_pipeline.upload()

        upload_count = session.get_data_table(self.table_name, self.job_id).count()
        return upload_count == self.data_count

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
