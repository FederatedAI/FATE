import click
from fate.components.entrypoint.cli.test import execute

test = click.Group(name="test")
test.add_command(execute.execute)

if __name__ == "__main__":
    test(prog_name="python -m fate.components.entrypoint.cli.test")
