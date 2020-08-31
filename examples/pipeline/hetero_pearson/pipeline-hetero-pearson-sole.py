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

from pipeline.component.hetero_pearson import HeteroPearson

from ._common_component import run_pipeline, get_config


def main(config="../../config.yaml", namespace=""):
    if isinstance(config, str):
        config = Config.load(config)
    hetero_pearson = HeteroPearson(name="hetero_pearson_0", column_indexes=-1, cross_parties=False)
    pipeline = run_pipeline(config=config,
                            guest_data={"name": "breast_hetero_guest", "namespace": "experiment"},
                            host_data={"name": "breast_hetero_host", "namespace": "experiment"},
                            hetero_pearson=hetero_pearson,
                            namespace=namespace)
    print(pipeline.get_component("hetero_pearson_0").get_model_param())
    print(pipeline.get_component("hetero_pearson_0").get_summary())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("-config", type=str,
                        help="config file")
    args = parser.parse_args()
    if args.config is not None:
        main(args.config)
    else:
        main()
