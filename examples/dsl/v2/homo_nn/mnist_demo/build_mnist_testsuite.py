import json

indent = 2


def build_model():
    from tensorflow.keras.layers import Conv2D, MaxPooling2D
    from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape
    from tensorflow.keras.models import Sequential

    model = Sequential()
    model.add(Reshape((28, 28, 1)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation="softmax"))
    return model


def main():
    m = build_model()
    nn_define = m.to_json()
    build_conf(nn_define)
    build_predict_conf()
    build_dsl()
    build_predict_dsl()
    build_testsuite()


def build_conf(nn_define):
    nn_define_dict = json.loads(nn_define)
    conf_dict = {
        "initiator": {"role": "guest", "party_id": 9999},
        "role": {"guest": [10000], "host": [10000], "arbiter": [10000]},
        "job_parameters": {"common": {"work_mode": 1, "backend": 0, "dsl_version": 2}},
        "component_parameters": {
            "role": {
                "host": {
                    "0": {
                        "reader_0": {
                            "table": {"name": "mnist_train", "namespace": "experiment"},
                        },
                        "dataio_0": {
                            "with_label": True,
                            "label_name": "label",
                            "label_type": "int",
                            "output_format": "dense",
                        },
                    }
                },
                "guest": {
                    "0": {
                        "reader_0": {
                            "table": {"name": "mnist_train", "namespace": "experiment"}
                        },
                        "dataio_0": {
                            "with_label": True,
                            "label_name": "label",
                            "label_type": "int",
                            "output_format": "dense",
                        },
                    }
                },
            },
            "common": {
                "dataio_0": {"with_label": True},
                "homo_nn_0": {
                    "encode_label": True,
                    "max_iter": 3,
                    "batch_size": 128,
                    "early_stop": {"early_stop": "diff", "eps": 0.0001},
                    "optimizer": {"learning_rate": 0.001, "optimizer": "Adam"},
                    "loss": "categorical_crossentropy",
                    "metrics": ["accuracy"],
                    "nn_define": nn_define_dict,
                    "config_type": "keras",
                },
                "evaluation_0": {"eval_type": "multi"},
            },
        },
    }
    with open("mnist_conf.json", "w") as f:
        json.dump(conf_dict, f, indent=indent)


def build_dsl():
    dsl_dict = {
        "components": {
            "reader_0": {"module": "Reader", "output": {"data": ["data"]}},
            "dataio_0": {
                "module": "DataIO",
                "input": {"data": {"data": ["reader_0.data"]}},
                "output": {"data": ["data"], "model": ["model"]},
            },
            "homo_nn_0": {
                "module": "HomoNN",
                "input": {"data": {"train_data": ["dataio_0.data"]}},
                "output": {"data": ["data"], "model": ["model"]},
            },
            "evaluation_0": {
                "module": "Evaluation",
                "input": {"data": {"data": ["homo_nn_0.data"]}},
                "output": {"data": ["data"]},
            },
        }
    }
    with open("mnist_dsl.json", "w") as f:
        json.dump(dsl_dict, f, indent=indent)


def build_predict_conf():
    conf_dict = {
        "initiator": {"role": "guest", "party_id": 9999},
        "role": {"guest": [10000], "host": [10000], "arbiter": [10000]},
        "job_parameters": {
            "work_mode": 1,
            "backend": 0,
            "dsl_version": 2,
            "job_type": "predict",
            "model_id": "arbiter-10000#guest-9999#host-10000#model",
            "model_version": "",
        },
        "role_parameters": {
            "guest": {
                "0": {
                    "reader_0": {
                        "table": {"name": "mnist_test", "namespace": "experiment"}
                    }
                }
            },
            "host": {
                "0": {
                    "reader_0": {
                        "table": {"name": "mnist_test", "namespace": "experiment"}
                    }
                }
            },
        },
    }
    with open("mnist_predict_conf.json", "w") as f:
        json.dump(conf_dict, f, indent=indent)


def build_predict_dsl():
    dsl_dict = {
        "components": {
            "reader_0": {"module": "Reader", "output": {"data": ["data"]}},
            "dataio_0": {
                "input": {
                    "data": {"data": ["reader_0.data"]},
                    "model": ["pipeline.dataio_0.model"],
                },
                "module": "DataIO",
                "output": {"data": ["data"]},
            },
            "homo_nn_0": {
                "input": {
                    "data": {"test_data": ["dataio_0.data"]},
                    "model": ["pipeline.homo_nn_0.model"],
                },
                "module": "HomoNN",
                "output": {"data": ["data"]},
            },
            "evaluation_0": {
                "input": {"data": {"data": ["homo_nn_0.data"]}},
                "module": "Evaluation",
                "output": {"data": ["data"]},
            },
        }
    }
    with open("mnist_predict_dsl.json", "w") as f:
        json.dump(dsl_dict, f, indent=indent)


def build_testsuite():
    testsuite_dict = {
        "data": [
            {
                "file": "examples/data/mnist_test.csv",
                "head": 1,
                "partition": 16,
                "table_name": "mnist_test",
                "namespace": "experiment",
                "role": "guest_0",
            },
            {
                "file": "examples/data/mnist_test.csv",
                "head": 1,
                "partition": 16,
                "table_name": "mnist_test",
                "namespace": "experiment",
                "role": "host_0",
            },
            {
                "file": "examples/data/mnist_train.csv",
                "head": 1,
                "partition": 16,
                "table_name": "mnist_train",
                "namespace": "experiment",
                "role": "guest_0",
            },
            {
                "file": "examples/data/mnist_train.csv",
                "head": 1,
                "partition": 16,
                "table_name": "mnist_train",
                "namespace": "experiment",
                "role": "host_0",
            },
        ],
        "tasks": {
            "mnist_conv": {"conf": "./mnist_conf.json", "dsl": "./mnist_dsl.json"},
            "mnist_predict": {
                "deps": "mnist_conv",
                "conf": "./mnist_predict_conf.json",
                "dsl": "./mnist_predict_dsl.json",
            },
        },
    }
    with open("mnist_testsuite.json", "w") as f:
        json.dump(testsuite_dict, f, indent=indent)


if __name__ == "__main__":
    main()
