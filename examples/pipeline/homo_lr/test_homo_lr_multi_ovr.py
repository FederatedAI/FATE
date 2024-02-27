import argparse
from fate_client.pipeline.components.fate import HomoLR, Evaluation, Reader
from fate_client.pipeline import FateFlowPipeline
from fate_client.pipeline.utils import test_utils


def main(config="../config.yaml", namespace=""):
    if isinstance(config, str):
        config = test_utils.load_job_config(config)
    parties = config.parties
    guest = parties.guest[0]
    host = parties.host[0]
    arbiter = parties.arbiter[0]
    pipeline = FateFlowPipeline().set_parties(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader("reader_0", runtime_parties=dict(guest=guest, host=host))
    reader_0.guest.task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_homo_guest"
    )
    reader_0.hosts[[0, 1]].task_parameters(
        namespace=f"experiment{namespace}",
        name="breast_homo_host"
    )
    homo_lr_0 = HomoLR(
        "homo_lr_0",
        epochs=10,
        batch_size=16,
        ovr=True,
        label_num=4,
        train_data=reader_0.outputs["output_data"]
    )

    evaluation_0 = Evaluation(
        'eval_0',
        default_eval_setting='multi',
        input_datas=[homo_lr_0.outputs['train_output_data']]
    )

    pipeline.add_tasks([reader_0, homo_lr_0, evaluation_0])
    pipeline.compile()
    pipeline.fit()
    print (pipeline.get_task_info("homo_lr_0").get_output_data())
    print(pipeline.get_task_info("homo_lr_0").get_output_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser("PIPELINE DEMO")
    parser.add_argument("--config", type=str, default="../config.yaml",
                        help="config file")
    parser.add_argument("--namespace", type=str, default="",
                        help="namespace for data stored in FATE")
    args = parser.parse_args()
    main(config=args.config, namespace=args.namespace)