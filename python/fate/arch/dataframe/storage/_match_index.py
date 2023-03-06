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

"""
This Structure is used for psi process, support very limited operation: like hash/curve25519
"""

class MatchIndex(object):
    def __init__(self, ctx, match_index_table):
        """
        match_index_table: each partition is a pandas dataframe index object
        """
        self._ctx = ctx
        self._match_index_table = match_index_table
        self._count = None

    def count(self):
        if not self._count:
            self._count = self._match_index_table.count()

        return self._count
