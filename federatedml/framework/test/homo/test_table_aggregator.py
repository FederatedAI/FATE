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

from fate_arch.session import computing_session as session
from federatedml.framework.homo.procedure import table_aggregator
from federatedml.framework.test.homo.homo_test_sync_base import TestSyncBase
from federatedml.util import consts
from federatedml.util import LOGGER


class TableAggregatorTest(object):

    def init_table(self):
        final_result = [(i, i) for i in range(20)]

        table = session.parallelize(final_result,
                                    include_key=True,
                                    partition=10)
        return table

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        agg = aggregator.with_role(role, transfer_variable, enable_secure_aggregate=True)
        LOGGER.debug(f"Finish with_role, agg: {agg}")
        if role == consts.ARBITER:
            LOGGER.debug(f"role: {role}, start")
            agg.aggregate_and_broadcast()
            LOGGER.debug(agg.aggregate_loss())
        else:
            # disorder dit
            # order = list(range(5))
            # np.random.seed(random.SystemRandom().randint(1, 100))
            # np.random.shuffle(order)
            # raw = {k: np.random.rand(10, 10) for k in order}
            #
            # w = OrderDictWeights(copy.deepcopy(raw))
            # d = random.random()
            LOGGER.debug(f"role: {role}, start")

            final_result = [(i, i) for i in range(20)]

            table = session.parallelize(final_result,
                                        include_key=True,
                                        partition=10)
            LOGGER.debug(f"role: {role}, start aggregate_then_get")

            aggregated = agg.aggregate_then_get(table)
            LOGGER.debug(f"role: {role}, finish aggregate_then_get")


            return aggregated

    def run_with_num_hosts(self, num_hosts):
        guest_table = self.init_table()


        _, guest, *hosts = self.run_results(num_hosts)

        expected = [(i, i * num_hosts) for i in range(20)]

        aggregated = [guest]
        for host in hosts:
            aggregated.append(host)

        for table in aggregated:
            result = sorted(list(table.collect()), key=lambda x: x[0])
            for k, r in enumerate(result):
                self.assertAlmostEqual(expected[k][1], r)

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
