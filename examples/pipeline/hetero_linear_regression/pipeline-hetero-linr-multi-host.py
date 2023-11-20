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

import argparse

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import Evaluation
from pipeline.component import HeteroLinR
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data

from pipeline.utils.tools import load_job_config


def main(config="../../config.yaml", namespace=""):
    # obtain config
    if isinstance(config, str):
        config = load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    hosts = parties.host
    arbiter = parties.arbiter[0]

    guest_train_data = {"name": "motor_hetero_guest", "namespace": f"experiment{namespace}"}
    host_train_data = [{"name": "motor_hetero_host", "namespace": f"experiment{namespace}"},
                       {"name": "motor_hetero_host", "namespace": f"experiment{namespace}"}]

    pipeline = PipeLine().set_initiator(
        role='guest', party_id=guest).set_roles(
        guest=guest, host=hosts, arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role='guest', party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role='host', party_id=hosts[0]).component_param(table=host_train_data[0])
    reader_0.get_party_instance(role='host', party_id=hosts[1]).component_param(table=host_train_data[1])

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(
        role='guest',
        party_id=guest).component_param(
        with_label=True,
        label_name="motor_speed",
        label_type="float",
        output_format="dense")
    data_transform_0.get_party_instance(role='host', party_id=hosts).component_param(with_label=False)

    intersection_0 = Intersection(
        name="intersection_0",
        intersect_method="rsa",
        rsa_params={"hash_method": "sha256", "final_hash_method": "sha256", "key_length": 1024})
    hetero_linr_0 = HeteroLinR(name="hetero_linr_0", penalty="L2", optimizer="sgd", tol=0.001,
                               alpha=0.01, max_iter=20, early_stop="weight_diff", batch_size=-1,
                               learning_rate=0.15, decay=0.0, decay_sqrt=False,
                               encrypt_param={"key_length": 1024},
                               init_param={"init_method": "zeros"})

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="regression", pos_label=1)
    # evaluation_0.get_party_instance(role='host', party_id=hosts[0]).component_param(need_run=False)
    # evaluation_0.get_party_instance(role='host', party_id=hosts[1]).component_param(need_run=False)

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_linr_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_linr_0.output.data))

    pipeline.compile()

    pipeline.fit()

    # predict
    # deploy required components
    pipeline.deploy_component([data_transform_0, intersection_0, hetero_linr_0])

    predict_pipeline = PipeLine()
    # add data reader onto predict pipeline
    predict_pipeline.add_component(reader_0)
    # add selected components from train pipeline onto predict pipeline
    # specify data source
    predict_pipeline.add_component(
        pipeline, data=Data(
            predict_input={
                pipeline.data_transform_0.input.data: reader_0.output.data}))
    # run predict model
    predict_pipeline.predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
