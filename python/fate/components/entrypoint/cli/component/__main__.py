import click
from fate.components.entrypoint.cli.component import (
    artifact_type_cli,
    cleanup_cli,
    desc_cli,
    execute_cli,
    list_cli,
    task_schema_cli,
)

component = click.Group(name="component")
component.add_command(execute_cli.execute)
component.add_command(cleanup_cli.cleanup)
component.add_command(desc_cli.desc)
component.add_command(list_cli.list)
component.add_command(artifact_type_cli.artifact_type)
component.add_command(task_schema_cli.task_schema)

if __name__ == "__main__":
    component(prog_name="python -m fate.components.entrypoint.cli.component")
