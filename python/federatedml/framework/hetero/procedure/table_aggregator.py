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

import numpy as np

from fate_arch.session import computing_session
from federatedml.framework.homo.blocks import aggregator
from federatedml.framework.homo.blocks import secure_mean_aggregator
from federatedml.framework.homo.blocks import secure_sum_aggregator
from federatedml.util import LOGGER
from federatedml.util import consts


class TableScatterTransVar(secure_sum_aggregator.AggregatorTransVar,
                           secure_mean_aggregator.SecureMeanAggregatorTransVar):
    def __init__(self, server=(consts.ARBITER,), clients=(consts.GUEST, consts.HOST), prefix=None):
        super().__init__(server=server, clients=clients, prefix=prefix)
        self.client_table = self.create_client_to_server_variable(name="client_table")
        self.server_table = self.create_server_to_client_variable(name="server_table")


class TableTransferServer(object):

    def __init__(self, trans_var: TableScatterTransVar = None):
        if trans_var is None:
            trans_var = TableScatterTransVar()
        self._scatter = trans_var.client_table
        self._broadcaster = trans_var.server_table
        self._client_parties = trans_var.client_parties

    def get_tables(self, suffix=tuple()):
        tables = self._scatter.get_parties(parties=self._client_parties, suffix=suffix)
        return tables

    def send_tables(self, tables, parties=None, suffix=tuple()):
        parties = self._client_parties if parties is None else parties
        LOGGER.debug("In TableTransferServer, send_tables")
        return self._broadcaster.remote_parties(obj=tables, parties=parties, suffix=suffix)

    @property
    def client_parties(self):
        return self._client_parties


class TableTransferClient(object):

    def __init__(self, trans_var: TableScatterTransVar = None):
        if trans_var is None:
            trans_var = TableScatterTransVar()
        self._scatter = trans_var.client_table
        self._broadcaster = trans_var.server_table
        self._server_parties = trans_var.server_parties

    def get_tables(self, suffix=tuple()):
        return self._broadcaster.get_parties(parties=self._server_parties, suffix=suffix)[0]

    def send_tables(self, tables, suffix=tuple()):
        return self._scatter.remote_parties(obj=tables, parties=self._server_parties, suffix=suffix)


class Server(secure_sum_aggregator.Server, secure_mean_aggregator.Server):

    def __init__(self, trans_var: TableScatterTransVar = None, enable_secure_aggregate=True):
        if trans_var is None:
            trans_var = TableScatterTransVar()
        super().__init__(trans_var=trans_var, enable_secure_aggregate=enable_secure_aggregate)

        self.enable_secure_aggregate = enable_secure_aggregate
        # if enable_secure_aggregate:
        # table_random_padding_cipher.Server(trans_var=trans_var.random_padding_cipher_trans_var) \
        #     .exchange_secret_keys()
        self._table_sync = TableTransferServer()

    def aggregate_tables(self, suffix=tuple()):
        tables = self._table_sync.get_tables(suffix=suffix)
        result = tables[0]
        for table in tables[1:]:
            result = result.join(table, lambda x1, x2: x1 + x2)
        # LOGGER.debug(f"aggregate_result: {list(result.collect())[0]}")
        return result

    def send_aggregated_tables(self, table, suffix=tuple()):
        for party in self._table_sync.client_parties:
            self._table_sync.send_tables(tables=table, parties=party, suffix=suffix)

    def aggregate_and_broadcast_table(self, suffix=tuple()):
        """
        aggregate tables from guest and hosts, then broadcast the aggregated table.

        Args:
            suffix: tag suffix
        """
        LOGGER.debug(f"Start aggregate_and_broadcast")
        table = self.aggregate_tables(suffix=suffix)
        self.send_aggregated_tables(table, suffix=suffix)
        return table


class Client(secure_sum_aggregator.Client, secure_mean_aggregator.Client):

    def __init__(self, trans_var: TableScatterTransVar = None, enable_secure_aggregate=True):
        if trans_var is None:
            trans_var = TableScatterTransVar()
        super().__init__(trans_var, enable_secure_aggregate)
        self._table_sync = TableTransferClient()

        self.enable_secure_aggregate = enable_secure_aggregate

        if enable_secure_aggregate:
            # self._table_random_padding_cipher = table_random_padding_cipher.Client(
            #     trans_var=trans_var.random_padding_cipher_trans_var).create_cipher()
            self._random_padding_cipher.set_amplify_factor(consts.SECURE_AGG_AMPLIFY_FACTOR)

    def secure_aggregate_table(self, send_func, table, enable_secure_aggregate=True):
        """
        Secure aggregate tables.

        Degree is useless, exist for extension.
        """
        LOGGER.debug(f"In secure aggregate, enable_secure_aggregate: {enable_secure_aggregate}")
        if enable_secure_aggregate:
            table = self._random_padding_cipher.encrypt_table(table)
            LOGGER.debug("Finish add random numbers")

        send_func(table)

    def send_table(self, table, suffix=tuple()):
        def _func(_table):
            # LOGGER.debug(f"cipher table content: {list(_table.collect())[0]}")
            self._table_sync.send_tables(_table, suffix=suffix)

        # LOGGER.debug(f"plantext table content: {list(table.collect())[0]}")

        return self.secure_aggregate_table(send_func=_func,
                                           table=table,
                                           enable_secure_aggregate=self.enable_secure_aggregate)

    def get_aggregated_table(self, suffix=tuple()):
        return self._table_sync.get_tables(suffix=suffix)

    def aggregate_then_get_table(self, table, suffix=tuple()):
        self.send_table(table=table, suffix=suffix)
        return self.get_aggregated_table(suffix=suffix)
