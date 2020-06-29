import argparse
import json

import numpy as np
import pandas as pd

GUEST = "guest"
HOST = "host"


def create_conf(name, pairs):
    for r, c in pairs:
        sub = f"{name}_{r}_{c}"
        conf = {
            "initiator": {"role": "guest", "party_id": 10000},
            "job_parameters": {"work_mode": 0},
            "role": {"guest": [10000], "host": [10000]},
            "role_parameters": {
                role: {
                    "args": {"data": {"data": [{"name": f"{sub}_{role}", "namespace": "experiment"}]}},
                    "dataio_0": {"with_label": [False], "output_format": ["dense"]},
                    "pearson_0": {"column_indexes": [-1]}
                } for role in [GUEST, HOST]
            }
        }
        with open(f"{sub}_conf.json", "w") as f:
            json.dump(conf, f, indent=2)


def create_test_suite(name, pairs):
    def data_pair(sub_name):
        return [{
            "file": f"examples/federatedml-1.x-examples/hetero_pearson/{sub_name}_{role}.csv",
            "head": 1,
            "partition": 16,
            "work_mode": 0,
            "table_name": f"{sub_name}_{role}",
            "namespace": "experiment",
            "role": f"{role}_0"
        } for role in [GUEST, HOST]]

    data = []
    task = {}
    for r, c in pairs:
        sub = f"{name}_{r}_{c}"
        data.extend(data_pair(sub))
        task[f"pearson_{sub}"] = {"conf": f"./{sub}_conf.json",
                                  "dsl": "./test_dsl.json"}

    with open(f"{name}_testsuite.json", "w") as f:
        json.dump({"data": data, "tasks": task}, f, indent=2)


def create_data(role, name, pairs):
    for r, c in pairs:
        sub = f"{name}_{r}_{c}"
        arr = np.random.rand(r, c)
        df = pd.DataFrame(arr)
        df.to_csv(f"{sub}_{role}.csv", index_label="id", float_format="%.04f")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", type=str, choices=[GUEST, HOST])
    parser.add_argument("name", type=str)
    parser.add_argument("shapes", nargs="*", type=int)
    args = parser.parse_args()

    if len(args.shapes) % 2 == 1:
        print("error: num of shape is odd number, exit.")
        return

    pairs = []
    print(f"data shapes:")
    for i in range(len(args.shapes) // 2):
        pair = (args.shapes[2 * i], args.shapes[2 * i + 1])
        pairs.append(pair)
        print(pair)

    if args.role.lower() == GUEST:
        create_conf(args.name, pairs)
        create_test_suite(args.name, pairs)
    create_data(args.role.lower(), args.name, pairs)


if __name__ == '__main__':
    main()
