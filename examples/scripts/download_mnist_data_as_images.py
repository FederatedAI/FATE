#!/usr/bin/env python3

import os
from pathlib import Path
from ruamel import yaml
import torchvision


def download_mnist(base, name, is_train=True):
    dataset = torchvision.datasets.MNIST(
        root=base.joinpath(".cache"), train=is_train, download=True
    )
    converted_path = base.joinpath(name)
    converted_path.mkdir(exist_ok=True)

    inputs_path = converted_path.joinpath("images")
    inputs_path.mkdir(exist_ok=True)
    targets_path = converted_path.joinpath("targets")
    config_path = converted_path.joinpath("config.yaml")
    filenames_path = converted_path.joinpath("filenames")

    with filenames_path.open("w") as filenames:
        with targets_path.open("w") as targets:
            for idx, (img, target) in enumerate(dataset):
                filename = f"{idx:05d}"

                # save img
                img.save(inputs_path.joinpath(f"{filename}.jpg"))

                # save target
                targets.write(f"{filename},{target}\n")

                # save filenames
                filenames.write(f"{filename}\n")

    config = {
        "type": "vision",
        "inputs": {"type": "images", "ext": "jpg", "PIL_mode": "L"},
        "targets": {"type": "integer"},
    }
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, indent=2, default_flow_style=False)


def main():
    data_dir = os.path.realpath(
        os.path.join(os.path.realpath(__file__), os.path.pardir, os.path.pardir, "data")
    )
    download_mnist(Path(data_dir), "mnist_train")
    download_mnist(Path(data_dir), "mnist_eval", is_train=False)


if __name__ == "__main__":
    main()
