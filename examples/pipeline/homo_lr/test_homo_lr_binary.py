import argparse
from fate_client.pipeline.components.fate import HomoLR, Evaluation
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.interface import DataWarehouseChannel
from fate_client.pipeline.utils import test_utils


def main(config="../config.yaml", namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]
    pipeline = FateFlowPipeline().set_parties(guest=guest, host=host, arbiter=arbiter)

    homo_lr_0 = HomoLR(
        "homo_lr_0",
        epochs=10,
        batch_size=16
    )

    homo_lr_0.guest.task_setting(train_data=DataWarehouseChannel(name="breast_homo_guest", namespace="experiment"))
    homo_lr_0.hosts[0].task_setting(train_data=DataWarehouseChannel(name="breast_homo_host", namespace="experiment"))
    evaluation_0 = Evaluation(
        'eval_0',
        metrics=['auc'],
        input_data=[homo_lr_0.outputs['train_output_data']]
    )


    pipeline.add_task(homo_lr_0)
    pipeline.add_task(evaluation_0)
    pipeline.compile()
    pipeline.fit()
    print (pipeline.get_task_info("homo_lr_0").get_output_data())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)