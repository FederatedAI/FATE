import argparse
import json
from arch.api import eggroll
from arch.task_manager.adapter.offline_feature.get_feature import GetFeature


def do_import_feature(job_id, config_data):
    work_mode = config_data.get("work_mode")
    eggroll.init(job_id, work_mode)
    response = GetFeature.import_data(config_data)
    print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', '--job_id', required=True, type=str, help="job id to use")
    parser.add_argument('-c', '--config', required=True, type=str, help="you should provide a path of configure file with json format")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config_data = json.load(f)

        do_import_feature(args.job_id, config_data)
    except:
        raise ValueError("export file failed")
