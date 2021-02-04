#!/usr/bin/env python3

from pathlib import Path
from ruamel import yaml
import torchvision


def download_mnist(base=Path("./files")):
    dataset = torchvision.datasets.MNIST(
        root=base.joinpath(".cache"), train=True, download=True
    )
    converted_path = base.joinpath("converted")
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
        "inputs": {"type": "images", "ext": "jpg"},
        "target": {"type": "integer"},
    }
    with config_path.open("w") as f:
        yaml.safe_dump(config, f, indent=2)


if __name__ == "__main__":
    download_mnist()
