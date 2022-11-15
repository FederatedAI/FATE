from fate.components.spec import artifacts, roles, stages

from ..cpn import Cpn

cpn = Cpn(
    name="lr",
    roles=roles.get_all(),
    provider="fate",
    version="2.0.0.alpha",
    description="",
)


@cpn.artifact("train_data", type=artifacts.TrainData, roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN])
@cpn.artifact(
    "validate_data",
    type=artifacts.ValidateData,
    optional=True,
    roles=[roles.GUEST, roles.HOST],
    stages=["train"],
)
@cpn.artifact("input_model", type=artifacts.Model, roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN])
@cpn.artifact(
    "test_data",
    type=artifacts.TestData,
    optional=True,
    roles=[roles.GUEST, roles.HOST],
    stages=[stages.PREDICT],
)
@cpn.parameter("learning_rate", type=float, default=0.1, optional=False)
@cpn.parameter("max_iter", type=int, default=100, optional=False)
@cpn.artifact(
    "train_output_data",
    type=artifacts.TrainOutputData,
    roles=[roles.GUEST, roles.HOST],
    stages=[stages.TRAIN],
)
@cpn.artifact("output_model", type=artifacts.Model, roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN])
@cpn.artifact(
    "test_output_data",
    type=artifacts.TestOutputData,
    roles=[roles.GUEST, roles.HOST],
    stages=[stages.PREDICT],
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
    print(cpn.dump_yaml())
