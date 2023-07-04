import click


@click.command()
@click.option("--name", type=str, required=True, help="component name")
@click.option("--role", type=str, required=True, help="component name")
@click.option("--stage", type=str, required=True, help="component name")
@click.option("--output-path", type=click.File("w", lazy=True), help="output path")
def artifact_type(name, role, stage, output_path):
    from fate.components.core import Role, Stage, load_component

    cpn = load_component(name)
    role = Role.from_str(role)
    stage = Stage.from_str(stage)
    if not stage.is_default:
        for stage_component in cpn.stage_components:
            if stage_component.name == stage.name:
                cpn = stage_component
                break
        else:
            supported_stage_names = [stage_component.name for stage_component in cpn.stage_components]
            raise ValueError(f"stage `{stage.name}` not in supported stage names: {supported_stage_names}")

    if output_path:
        cpn.dump_runtime_io_yaml(role, stage, output_path)
    else:
        print(cpn.dump_runtime_io_yaml(role, stage, output_path))


if __name__ == "__main__":
    artifact_type()
