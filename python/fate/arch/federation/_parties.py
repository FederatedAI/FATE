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


class Party:
    """
    Uniquely identify
    """

    def __init__(self, role, party_id):
        self.role = str(role)
        self.party_id = str(party_id)

    def __hash__(self):
        return (self.role, self.party_id).__hash__()

    def __str__(self):
        return f"Party(role={self.role}, party_id={self.party_id})"

    def __repr__(self):
        return self.__str__()

    def __lt__(self, other):
        return (self.role, self.party_id) < (other.role, other.party_id)

    def __eq__(self, other):
        return self.party_id == other.party_id and self.role == other.role

    def as_tuple(self):
        return (self.role, self.party_id)
