from fate.components.specs.spec import (
    Cpn,
    Model,
    TestOutputData,
    TrainData,
    TrainOutputData,
    ValidateData,
)

cpn = Cpn(
    name="lr",
    roles=["guest", "host", "arbiter"],
    provider="fate",
    version="2.0.0.alpha",
    description="",
)


@cpn.artifact("train_data", type=TrainData, roles=["guest", "host"], stages=["train"])
@cpn.artifact(
    "validate_data",
    type=ValidateData,
    optional=True,
    roles=["guest", "host"],
    stages=["train"],
)
@cpn.artifact("input_model", type=Model, roles=["guest", "host"], stages=["train"])
@cpn.artifact(
    "test_data",
    type=ValidateData,
    optional=True,
    roles=["guest", "host"],
    stages=["test"],
)
@cpn.parameter("learning_rate", type=float, default=0.1, optional=False)
@cpn.parameter("max_iter", type=int, default=100, optional=False)
@cpn.artifact(
    "train_output_data", type=TrainOutputData, roles=["guest", "host"], stages=["train"]
)
@cpn.artifact("output_model", type=Model, roles=["guest", "host"], stages=["train"])
@cpn.artifact(
    "test_output_data", type=TestOutputData, roles=["guest", "host"], stages=["test"]
)
def lr(
    ctx,
    train_data,
    validate_data,
    input_model,
    lr,
    max_iter,
    train_output_data,
    output_model,
    test_output_data,
):
    from fate.ml.lr.guest import LrModuleGuest
    from fate.ml.lr.host import LrModuleHost

    def guest_run(stage="train"):
        if stage == "train":
            module = LrModuleGuest(lr=lr, max_iter=max_iter)
            module.fit(ctx, ctx.read_df(train_data), validate_data)
            model = module.export_model()
            output_data = module.predict(ctx, ctx.read_df(validate_data))
            return {
                "train_output_model": ctx.dump(output_data, train_output_data),
                "output_model": ctx.dump(output_model, model),
            }
        if stage == "predict":
            module = LrModuleGuest.load_model(ctx, ctx.load(input_model))
            output_data = module.predict(ctx, ctx.read_df(train_data))
            return {
                "test_output_data": ctx.dump(output_data, test_output_data),
            }
        else:
            raise NotImplementedError(f"stage={stage}")

    @auto_impl_stages(ctx, module=LrModuleHost, stages=["train", "predict"])
    def host_run(stage="train"):
        ...

    def arbiter_run(stage="train"):
        ...

    return dict(guest=guest_run, host=host_run, arbiter=arbiter_run)


if __name__ == "__main__":
    from ruamel import yaml

    print(yaml.dump(cpn.get_spec().dict()))
