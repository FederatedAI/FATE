import json

indent = 2


def build_model():
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Reshape((28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    return model


def main():
    m = build_model()
    nn_define = m.to_json()
    build_conf(nn_define)
    build_dsl()
    build_testsuite()


def build_conf(nn_define):
    nn_define_dict = json.loads(nn_define)
    conf_dict = {
        "initiator": {"role": "guest", "party_id": 10000},
        "job_parameters": {"work_mode": 0},
        "role": {"guest": [10000], "host": [10000], "arbiter": [10000]},
        "role_parameters": {
            "guest": {
                "args": {
                    "data":
                        {
                            "train_data": [{"name": "mnist_train", "namespace": "experiment"}],
                            "eval_data": [{"name": "mnist_test", "namespace": "experiment"}]
                        }
                },
                "dataio_0": {
                    "with_label": [True],
                    "label_name": ["label"],
                    "label_type": ["int"],
                    "output_format": ["dense"]
                },
                "dataio_1": {
                    "with_label": [True],
                    "label_name": ["label"],
                    "label_type": ["int"],
                    "output_format": ["dense"]
                }
            },
            "host": {
                "args": {
                    "data": {
                        "train_data": [{"name": "mnist_train", "namespace": "experiment"}],
                        "eval_data": [{"name": "mnist_test", "namespace": "experiment"}]
                    }
                },
                "dataio_0": {
                    "with_label": [True],
                    "label_name": ["label"],
                    "label_type": ["int"],
                    "output_format": ["dense"]
                },
                "dataio_1": {
                    "with_label": [True],
                    "label_name": ["label"],
                    "label_type": ["int"],
                    "output_format": ["dense"]
                }
            }
        },
        "algorithm_parameters": {
            "homo_nn_0": {
                "config_type": "keras",
                "nn_define": nn_define_dict,
                "batch_size": 128,
                "optimizer": {"optimizer": "Adam", "learning_rate": 0.001},
                "early_stop": {"early_stop": "diff", "eps": 1e-4},
                "loss": "categorical_crossentropy",
                "metrics": ["accuracy", "AUC"],
                "max_iter": 3,
                "encode_label": True,
                "aggregate_every_n_epoch": 2
            },
            "evaluation_0": {
                "eval_type": "multi"
            }
        }
    }
    with open("mnist_conf.json", "w") as f:
        json.dump(conf_dict, f, indent=indent)


def build_testsuite():
    testsuite_dict = {
        "data": [
            {
                "file": "examples/data/mnist_test.csv",
                "head": 1,
                "partition": 16,
                "table_name": "mnist_test",
                "namespace": "experiment",
                "role": "guest_0"
            },
            {
                "file": "examples/data/mnist_test.csv",
                "head": 1,
                "partition": 16,
                "table_name": "mnist_test",
                "namespace": "experiment",
                "role": "host_0"
            },
            {
                "file": "examples/data/mnist_train.csv",
                "head": 1,
                "partition": 16,
                "table_name": "mnist_train",
                "namespace": "experiment",
                "role": "guest_0"
            },
            {
                "file": "examples/data/mnist_train.csv",
                "head": 1,
                "partition": 16,
                "table_name": "mnist_train",
                "namespace": "experiment",
                "role": "host_0"
            }
        ],
        "tasks": {
            "mnist_conv": {
                "conf": "./mnist_conf.json",
                "dsl": "./mnist_dsl.json"
            }
        }
    }
    with open("mnist_testsuite.json", "w") as f:
        json.dump(testsuite_dict, f, indent=indent)


def build_dsl():
    dsl_dict = {
        "components": {
            "dataio_0": {
                "module": "DataIO",
                "input": {"data": {"data": ["args.train_data"]}},
                "output": {"data": ["train"], "model": ["dataio"]}
            },
            "dataio_1": {
                "module": "DataIO",
                "input": {"data": {"data": ["args.eval_data"]}},
                "output": {"data": ["eval"], "model": ["dataio"]}
            },
            "homo_nn_0": {
                "module": "HomoNN",
                "input": {"data": {"train_data": ["dataio_0.train"]}},
                "output": {"data": ["train"], "model": ["homo_nn"]}
            },
            "homo_nn_1": {
                "module": "HomoNN",
                "input": {"data": {"eval_data": ["dataio_1.eval"]}, "model": ["homo_nn_0.homo_nn"]},
                "output": {"data": ["eval"], "model": ["homo_nn2"]}
            },
            "evaluation_0": {
                "module": "Evaluation",
                "input": {"data": {"data": ["homo_nn_1.eval"]}}
            }
        }
    }
    with open("mnist_dsl.json", "w") as f:
        json.dump(dsl_dict, f, indent=indent)


if __name__ == '__main__':
    main()
