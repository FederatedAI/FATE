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

from federatedml.util import consts


class Arbiter:
    """
    Arbiter for user ids transfer sync.
    """
    def register_user_ids_transfer(self, transfer_variable):
        """
        Not implemented here.
        """
        pass


class Client:
    """
    Client for user ids transfer sync.
    """
    def __init__(self):
        self.host_user_ids_sync = None
        self.guest_user_ids_sync = None

    def register_user_ids_transfer(self, transfer_variable):

        """
        Register user id transfer variables.
        :param transfer_variable:
        :return:
        """
        self.host_user_ids_sync = transfer_variable.host_user_ids
        self.guest_user_ids_sync = transfer_variable.guest_user_ids
        return self


class Host(Client):
    """
    Host for user ids transfer sync.
    """
    def send_host_user_ids(self, user_ids, suffix=tuple()):
        """
        Send user ids to guest.
        :param user_ids:
        :param suffix:
        :return:
        """
        self.host_user_ids_sync.remote(obj=user_ids, role=consts.GUEST, idx=0, suffix=suffix)

    def get_guest_user_ids(self, suffix=tuple()):
        """
        Receive user ids from guest.
        :param suffix:
        :return:
        """
        return self.guest_user_ids_sync.get(idx=0, suffix=suffix)


class Guest(Client):
    """
    Guest for user ids transfer sync.
    """
    def send_guest_user_ids(self, user_ids, suffix=tuple()):
        """
        Send user ids to host.
        :param user_ids:
        :param suffix:
        :return:
        """
        self.guest_user_ids_sync.remote(obj=user_ids, role=consts.HOST, idx=0, suffix=suffix)

    def get_host_user_ids(self, suffix=tuple()):
        """
        Receive user ids from host.
        :param suffix:
        :return:
        """
        host_user_ids = self.host_user_ids_sync.get(idx=0, suffix=suffix)
        return host_user_ids