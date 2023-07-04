import click


@click.command()
@click.option("--name", required=True, help="name of component_desc")
@click.option("--save", type=click.File(mode="w", lazy=True), help="save desc output to specified file in yaml format")
def desc(name, save):
    "generate component_desc describe config"
    from fate.components.core import load_component

    cpn = load_component(name)
    if save:
        cpn.dump_yaml(save)
    else:
        print(cpn.dump_yaml())


if __name__ == "__main__":
    desc()
