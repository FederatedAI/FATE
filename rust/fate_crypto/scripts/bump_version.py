import toml
import pathlib
import argparse

root_path = pathlib.Path(__file__).parent.parent.resolve()

def update_version(version):
    with open(root_path.joinpath("Cargo.toml")) as f:
        cargo = toml.load(f)
    old_version = cargo["package"]["version"]
    print(f"bump fate_crypto version from `{old_version}` to `{version}`")
    cargo["package"]["version"] = version
    with open(root_path.joinpath("Cargo.toml"), "w") as f:
        toml.dump(cargo, f)

if __name__ == "__main__":
    parse = argparse.ArgumentParser("bump version")
    parse.add_argument("version", type=str)
    args = parse.parse_args()
    update_version(args.version)
