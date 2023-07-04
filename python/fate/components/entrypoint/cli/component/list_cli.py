import click


@click.command()
@click.option("--save", type=click.File(mode="w", lazy=True), help="save list output to specified file in json format")
def list(save):
    "list all components"
    from fate.components.core import list_components

    if save:
        import json

        json.dump(list_components(), save)
    else:
        print(list_components())


if __name__ == "__main__":
    list()
