import click


@click.command()
@click.option("--save", type=click.File(mode="w", lazy=True), help="save desc output to specified file in yaml format")
def task_schema(save):
    "generate component_desc task config json schema"
    from fate.components.core.spec.task import TaskConfigSpec

    if save:
        save.write(TaskConfigSpec.schema_json())
    else:
        print(TaskConfigSpec.schema_json())


if __name__ == "__main__":
    task_schema()
