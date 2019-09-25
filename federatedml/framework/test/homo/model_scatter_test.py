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

from federatedml.framework.homo.sync import model_scatter_sync
from federatedml.util import consts
from .homo_test_sync_base import TestSyncBase
from federatedml.framework.weights import ListWeights
import random


class ModelScatterTest(TestSyncBase):

    @classmethod
    def call(cls, role, transfer_variable, ind, *args):
        cipher_dict = args[0]
        if role == consts.ARBITER:
            models = model_scatter_sync.Arbiter() \
                .register_model_scatter(transfer_variable.host_model,
                                        transfer_variable.guest_model)\
                .get_models(cipher_dict)
            return list(models)
        elif role == consts.HOST:
            model = [random.random() for _ in range(random.randint(1, 10))]
            if cipher_dict[ind]:
                model = cipher_dict[ind].encrypt_list(model)
            model = ListWeights(model)
            return model_scatter_sync.Host() \
                .register_model_scatter(transfer_variable.host_model)\
                .send_model(model.for_remote())
        else:
            model = ListWeights([random.random() for _ in range(random.randint(1, 10))])
            return model_scatter_sync.Guest() \
                .register_model_scatter(transfer_variable.guest_model)\
                .send_model(model.for_remote())

    def run_with_num_hosts(self, num_hosts):
        ratio = 0.3
        key_size = 1024

        import random
        from federatedml.secureprotol.encrypt import PaillierEncrypt
        PaillierEncrypt().generate_key(key_size)
        cipher_dict = {}
        for i in range(num_hosts):
            if random.random() > ratio:
                cipher = PaillierEncrypt()
                cipher.generate_key(key_size)
                cipher_dict[i] = cipher
            else:
                cipher_dict[i] = None

        arbiter, guest, *hosts = self.run_results(num_hosts, cipher_dict)

        arbiter = [x[0].unboxed for x in arbiter]
        guest = guest.unboxed
        hosts = [hosts[i].weights.decrypted(cipher_dict[i]).unboxed if cipher_dict[i] else hosts[i].unboxed
                 for i in range(num_hosts)]

        self.assertListEqual(arbiter[0], guest)
        for i in range(len(hosts)):
            self.assertListEqual(arbiter[i+1], hosts[i])

    def test_host_1(self):
        self.run_with_num_hosts(1)

    def test_host_10(self):
        self.run_with_num_hosts(10)
