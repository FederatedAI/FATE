# Develop Guide: Components

Starting in ver 2.0, FATE `components` serve as entry points to modules for job scheduler.
In general, computational logic should be contained within `ML modules`, while `components` call module functions with
proper input and pass respective result data and model to scheduler.

## Components CLI

Check available commands:

```commandline
python -m fate.components component --help
Usage: python -m fate.components component [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  artifact-type
  cleanup        cleanup
  desc           generate component_desc describe config
  execute        execute component
  list           list all components
  task-schema    generate component_desc task config json schema
```

List all components:

```commandline
python -m fate.components component list  
{'buildin': ['feature_scale', 'reader', 'coordinated_lr', 'coordinated_linr', 'homo_nn', 'hetero_nn', 'homo_lr', 'hetero_secureboost', 'dataframe_transformer', 'psi', 'evaluation', 'artifact_test', 'statistics', 'hetero_feature_binning', 'hetero_feature_selection', 'feature_correlation', 'union', 'sample', 'data_split', 'sshe_lr', 'sshe_linr', 'toy_example', 'dataframe_io_test', 'multi_model_test', 'cv_test2'], 'thirdparty': []}
```

To make description file for a new component:

```commandline
python -m fate.components component desc --name feature_scale --save feature_scale.yaml
```

If new component will be added to PipeLine, make sure to move the description file into
folder `$fate_client/python/fate_client/pipeline/component_define/fate` and include a python file that defines new
component in `${FATE-Client}/python/fate_client/pipeline/components`.

## New Component

A simple component example may be found [here](../../python/fate/components/components/feature_scale.py)
For more advanced case, please refer [here](../../python/fate/components/components/sshe_lr.py)

In general, follow the steps below to create a new component:

1. Create a new python file under `$fate_base/python/fate/components/components`. We recommend that file be named after
   new component.
2. Define the new component and add decorator `@cpn.component(roles=[$role], provider="{$provider_source}")` to
   component in the new python file. Include the component in loading list in `components/components/__init__.py`.
    ```python 
    @cpn.component(roles=[GUEST, HOST], provider="fate")
    def sshe_lr(ctx, role):
        ...
    ```

3. Implement component.

   If new component supports different stages(train, predict, and maybe cross validation), mark respective stage
   implementation with corresponding decorators:
    ```python
    @sshe_lr.train()
    def train():
       ...
  
    @sshe_lr.predict()
    def predict():
        ...
    ```

   Specify inputs and outputs for each stage implementation.

    ```python 
    @sshe_lr.train()
    def train(
       ctx: Context,
       role: Role,
       train_data: cpn.dataframe_input(roles=[GUEST, HOST]),
       validate_data: cpn.dataframe_input(roles=[GUEST, HOST], optional=True),
       epochs: cpn.parameter(type=params.conint(gt=0), default=20, desc="max iteration num"),
       batch_size: cpn.parameter(
           type=params.conint(ge=10),
           default=None,
           desc="batch size, None means full batch, otherwise should be no less than 10, default None",
       ),
       tol: cpn.parameter(type=params.confloat(ge=0), default=1e-4),
       early_stop: cpn.parameter(
           type=params.string_choice(["weight_diff", "diff", "abs"]),
           default="diff",
           desc="early stopping criterion, choose from {weight_diff, diff, abs}, if use weight_diff,"
           "weight will be revealed every epoch",
       ),
       learning_rate: cpn.parameter(type=params.confloat(ge=0), default=0.05, desc="learning rate"),
       reveal_every_epoch: cpn.parameter(
           type=bool, default=False, desc="whether reveal encrypted result every epoch, " "only accept False for now"
       ),
       init_param: cpn.parameter(
           type=params.init_param(),
           default=params.InitParam(method="random_uniform", fit_intercept=True, random_state=None),
           desc="Model param init setting.",
       ),
       threshold: cpn.parameter(
           type=params.confloat(ge=0.0, le=1.0), default=0.5, desc="predict threshold for binary data"
       ),
       reveal_loss_freq: cpn.parameter(
           type=params.conint(ge=1),
           default=1,
           desc="rounds to reveal training loss, " "only effective if `early_stop` is 'loss'",
       ),
       train_output_data: cpn.dataframe_output(roles=[GUEST]),
       output_model: cpn.json_model_output(roles=[GUEST, HOST]),
       warm_start_model: cpn.json_model_input(roles=[GUEST, HOST], optional=True)):
        train_ctx = ctx.sub_ctx("train")
       
    @sshe_lr.predict()
    def predict(
        ctx,
        role: Role,
        # threshold: cpn.parameter(type=params.confloat(ge=0.0, le=1.0), default=0.5),
        test_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        input_model: cpn.json_model_input(roles=[GUEST, HOST]),
        test_output_data: cpn.dataframe_output(roles=[GUEST]),
    ):
        predict_ctx = ctx.sub_ctx("predict")
   
    @sshe_lr.cross_validation()
    def cross_validation(
        ctx: Context,
       role: Role,
       cv_data: cpn.dataframe_input(roles=[GUEST, HOST]),
       epochs: cpn.parameter(type=params.conint(gt=0), default=20, desc="max iteration num"),
       batch_size: cpn.parameter(
           type=params.conint(ge=10),
           default=None,
           desc="batch size, None means full batch, otherwise should be no less than 10, default None",
       ),
       tol: cpn.parameter(type=params.confloat(ge=0), default=1e-4),
       early_stop: cpn.parameter(
           type=params.string_choice(["weight_diff", "diff", "abs"]),
           default="diff",
           desc="early stopping criterion, choose from {weight_diff, diff, abs}, if use weight_diff,"
           "weight will be revealed every epoch",
       ),
       learning_rate: cpn.parameter(type=params.confloat(ge=0), default=0.05, desc="learning rate"),
       init_param: cpn.parameter(
           type=params.init_param(),
           default=params.InitParam(method="random_uniform", fit_intercept=True, random_state=None),
           desc="Model param init setting.",
       ),
       threshold: cpn.parameter(
           type=params.confloat(ge=0.0, le=1.0), default=0.5, desc="predict threshold for binary data"
       ),
       reveal_every_epoch: cpn.parameter(
           type=bool, default=False, desc="whether reveal encrypted result every epoch, " "only accept False for now"
       ),
       reveal_loss_freq: cpn.parameter(
           type=params.conint(ge=1),
           default=1,
           desc="rounds to reveal training loss, " "only effective if `early_stop` is 'loss'",
       ),
       cv_param: cpn.parameter(
           type=params.cv_param(),
           default=params.CVParam(n_splits=5, shuffle=False, random_state=None),
           desc="cross validation param",
       ),
       metrics: cpn.parameter(type=params.metrics_param(), default=["auc"]),
       output_cv_data: cpn.parameter(type=bool, default=True, desc="whether output prediction result per cv fold"),
       cv_output_datas: cpn.dataframe_outputs(roles=[GUEST, HOST], optional=True),
    ):
        cv_ctx = ctx.sub_ctx("cross_validation")
    ```

   In the case where no differentiation between stages is needed,
   or that the component will always be executed with `default` stage, parameters may be directly defined in the
   component entry function:

    ```python 
    @cpn.component(roles=[GUEST, HOST], provider="fate")
    def data_split(
        ctx: Context,
        role: Role,
        input_data: cpn.dataframe_input(roles=[GUEST, HOST]),
        train_size: cpn.parameter(
            type=Union[params.confloat(ge=0.0, le=1.0), params.conint(ge=0)],
            default=None,
            desc="size of output training data, " "should be either int for exact sample size or float for fraction",
        ),
        validate_size: cpn.parameter(
            type=Union[params.confloat(ge=0.0, le=1.0), params.conint(ge=0)],
            default=None,
            desc="size of output validation data, " "should be either int for exact sample size or float for fraction",
        ),
        test_size: cpn.parameter(
            type=Union[params.confloat(ge=0.0, le=1.0), params.conint(ge=0)],
            default=None,
            desc="size of output test data, " "should be either int for exact sample size or float for fraction",
        ),
        stratified: cpn.parameter(
            type=bool,
            default=False,
            desc="whether sample with stratification, " "should not use this for data with continuous label values",
        ),
        random_state: cpn.parameter(type=params.conint(ge=0), default=None, desc="random state"),
        hetero_sync: cpn.parameter(
            type=bool,
            default=True,
            desc="whether guest sync data set sids with host, "
            "default True for hetero scenario, "
            "should set to False for local and homo scenario",
        ),
        train_output_data: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
        validate_output_data: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
        test_output_data: cpn.dataframe_output(roles=[GUEST, HOST], optional=True),
    ):
        ...
    ```

   Always include `context` and `role` in input list.

* `context` provides background and common tools for running modules, including recall and ports to send and receive
  data across parties.
    * For different stages, make sure to name and provide separate sub-context
* `role` corresponds to local party identity, useful when different roles execute distinct modular functions.

  In addition, components may have the following common types of inputs:

* dataframe: as `dataframe_input`, access data content by `read()`api
  ```python
  train_data_frame = train_data.read()
  columns = train_data_frame.schema.columns()
  ```
    * `train_data`: for train stage
    * `test_data`: for predict stage
    * `cv_data`: for cross validation
    * `input_data`: for default stage

* model: as `json_model_input`, access model dict by `read` api
  ```python
  model = input_model.read()
  ```
    * `input_model`: for predict stage
    * `warm_start_model`: for warm start
* parameter: as `cpn_parameter`
    * arbitrary values, should include type, default value, and short description

  For outputs, components may have these types:

* dataframe: as `dataframe_output`
    * `train_output_data`: for train stage
    * `predict_output_data`: for predict stage, or for `DataSplit` module
    * `validate_output_data`: for `DataSplit` module only
    * `output_cv_data`: for cross validation, usually optional
* model: as `json_model_output`
    * `train_output_model`: for train stage

  Substantiate output objects through `write` api:

    ```python
    model_dict = module.get_model()
    train_output_model.write(model_dict)

    data = module.transform(train_data)
    train_output_data.write(data)
    ```
  All inputs and outputs may be set as optional.

