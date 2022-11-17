from fate.components import cpn
from fate.components.spec import Input, ModelArtifact, Output, artifacts, roles, stages


@cpn.component(roles=roles.get_all(), provider="fate", version="2.0.0.alpha")
@cpn.artifact("train_data", type=artifacts.TrainData, roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN])
@cpn.artifact(
    "validate_data", type=artifacts.ValidateData, optional=True, roles=[roles.GUEST, roles.HOST], stages=["train"]
)
@cpn.artifact("input_model", type=Input[ModelArtifact], roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN])
@cpn.artifact(
    "test_data", type=artifacts.TestData, optional=True, roles=[roles.GUEST, roles.HOST], stages=[stages.PREDICT]
)
@cpn.parameter("learning_rate", type=float, default=0.1, optional=False)
@cpn.parameter("max_iter", type=int, default=100, optional=False)
@cpn.artifact(
    "train_output_data", type=artifacts.TrainOutputData, roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN]
)
@cpn.artifact("output_model", type=Output[ModelArtifact], roles=[roles.GUEST, roles.HOST], stages=[stages.TRAIN])
@cpn.artifact(
    "test_output_data", type=artifacts.TestOutputData, roles=[roles.GUEST, roles.HOST], stages=[stages.PREDICT]
)
def hetero_lr(
    ctx,
    role,
    stage,
    train_data,
    validate_data,
    test_data,
    input_model,
    learning_rate,
    max_iter,
    train_output_data,
    output_model,
    test_output_data,
):
    """ """
    from fate.ml.lr.guest import LrModuleGuest
    from fate.ml.lr.host import LrModuleHost

    if stage == "train":
        if role == "guest":
            module = LrModuleGuest(max_iter=max_iter)
            module.fit(ctx, ctx.read_df(train_data), validate_data)
            model = module.export_model()
            output_data = module.predict(ctx, ctx.read_df(validate_data))
            return {
                "train_output_model": ctx.dump(output_data, train_output_data),
                "output_model": ctx.dump(output_model, model),
            }

    elif stage == "predict":
        if role == "guest":
            module = LrModuleGuest.load_model(ctx, ctx.load(input_model))
            output_data = module.predict(ctx, ctx.read_df(train_data))
            return {
                "test_output_data": ctx.dump(output_data, test_output_data),
            }
    else:
        raise NotImplementedError(f"stage={stage}")


if __name__ == "__main__":
    print(hetero_lr.dump_yaml())
