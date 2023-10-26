from fate.components.components.nn.loader import Loader
from fate.components.components.nn.nn_runner import NNRunner
from fate.components.components.utils import consts
from fate.arch.dataframe import DataFrame
from fate.components.components.utils.tools import add_dataset_type
from fate.arch import Context
from fate.components.core import ARBITER, GUEST, HOST, Role, cpn
import logging
from fate.components.core.component_desc.artifacts.data._dataframe import DataframeReader, DataframeWriter
from fate.components.core.component_desc.artifacts.model._directory import ModelDirectoryReader, ModelDirectoryWriter


logger = logging.getLogger(__name__)


def prepare_runner_class(runner_module, runner_class, runner_conf, source):
    logger.info("runner conf is {}".format(runner_conf))
    logger.info("source is {}".format(source))
    if source is None:
        # load from default folder
        runner = Loader(
            "fate.components.components.nn.runner." +
            runner_module,
            runner_class,
            **runner_conf)()
    else:
        runner = Loader(
            runner_module,
            runner_class,
            source=source,
            **runner_conf)()
    assert isinstance(
        runner, NNRunner), "loaded class must be a subclass of NNRunner class, but got {}".format(
        type(runner))
    return runner


def prepare_context_and_role(runner, ctx, role, sub_ctx_name):
    sub_ctx = ctx.sub_ctx(sub_ctx_name)
    runner.set_context(sub_ctx)
    runner.set_role(role)


def get_input_data(stage, cpn_input_data):

    if stage == 'train':
        train_data, validate_data = cpn_input_data
        train_data = train_data.read()
        if validate_data is not None:
            validate_data = validate_data.read()

        return train_data, validate_data

    elif stage == 'predict':
        test_data = cpn_input_data
        test_data = test_data.read()
        return test_data
    else:
        raise ValueError(f"Unknown stage {stage}")


def prepared_saved_conf(
        model_conf,
        runner_class,
        runner_module,
        runner_conf,
        source):

    logger.info("loaded model_conf is: {}".format(model_conf))
    if "source" in model_conf:
        if source is None:
            source = model_conf["source"]

    runner_class_, runner_module_ = model_conf['runner_class'], model_conf['runner_module']
    if runner_class_ == runner_class and runner_module_ == runner_module:
        if "runner_conf" in model_conf:
            saved_conf = model_conf['runner_conf']
            saved_conf.update(runner_conf)
            runner_conf = saved_conf
            logger.info("runner_conf is updated: {}".format(runner_conf))
    else:
        logger.warning(
            "runner_class or runner_module is not equal to the saved model, "
            "use the new runner_conf, runner_class and runner module to train the model,\
                        saved module & class: {} {}, new module & class: {} {}".format(
                runner_module_, runner_class_, runner_module, runner_class))

    return runner_conf, source, runner_class, runner_module


def get_model_output_conf(runner_module,
                          runner_class,
                          runner_conf,
                          source,
                          ):
    return {
        "runner_module": runner_module,
        "runner_class": runner_class,
        "runner_conf": runner_conf,
        "source": source,
    }


def train_guest_and_host(
        ctx: Context,
        role: Role,
        train_data: DataframeReader,
        validate_data: DataframeReader,
        runner_module: str,
        runner_class: str,
        runner_conf: dict,
        source: str,
        train_data_output: DataframeWriter,
        train_model_output: ModelDirectoryWriter,
        train_model_input: ModelDirectoryReader
):

    if train_model_input is not None:
        model_conf = train_model_input.get_metadata()
        runner_conf, source, runner_class, runner_module = prepared_saved_conf(
            model_conf, runner_class, runner_module, runner_conf, source)
        saved_model_path = str(train_model_input.get_directory())
    else:
        saved_model_path = None

    runner: NNRunner = prepare_runner_class(
        runner_module, runner_class, runner_conf, source)
    prepare_context_and_role(runner, ctx, role, consts.TRAIN)

    output_dir = str(train_model_output.get_directory())
    train_data_, validate_data_ = get_input_data(
        consts.TRAIN, [train_data, validate_data])

    runner.train(train_data_, validate_data_, output_dir, saved_model_path)

    logger.info('Predicting Train & Validate Data')
    train_pred = runner.predict(train_data_, saved_model_path)
    if train_pred is not None:
        assert isinstance(
            train_pred, DataFrame), "train predict result should be a DataFrame"
        add_dataset_type(train_pred, consts.TRAIN_SET)

        if validate_data_ is not None:
            validate_pred = runner.predict(validate_data_)
            assert isinstance(
                validate_pred, DataFrame), "validate predict result should be a DataFrame"
            add_dataset_type(validate_pred, consts.VALIDATE_SET)
            output_df = DataFrame.vstack([train_pred, validate_pred])
        else:
            output_df = train_pred
        logger.info('write result dataframe')
        train_data_output.write(output_df)
    else:
        logger.warning(
            "train_pred is None, It seems that the runner is not able to predict. Failed to output data")

    output_conf = get_model_output_conf(runner_module,
                                        runner_class,
                                        runner_conf,
                                        source
                                        )
    train_model_output.write_metadata(output_conf)
