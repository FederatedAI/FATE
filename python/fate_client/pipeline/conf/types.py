class JobStage(object):
    TRAIN = "train"
    PREDICT = "predict"
    DEFAULT = "default"


class ArtifactSourceType(object):
    TASK_OUTPUT_ARTIFACT = "task_output_artifact"
    FATE_MODEL_WAREHOUSE = "fate_model_warehouse"


class ArtifactType(object):
    DATASET = "dataset"
    DATASETS = "datasets"
    MODEL = "model"
    MODELS = "models"
    METRIC = "metric"


class InputDataKeyType(object):
    TRAIN_DATA = "train_data"
    VALIDATE_DATA = "validate_data"
    TEST_DATA = "test_data"
    INPUT_DATA = "input_data"


class OutputDataKeyType(object):
    TRAIN_OUTPUT_DATA = "train_output_data"
    VALIDATE_OUTPUT_DATA = "validate_output_data"
    TEST_OUTPUT_DATA = "test_output_data"
    OUTPUT_DATA = "output_data"


class PlaceHolder(object):
    ...


class UriTypes(object):
    LOCAL = "file"
    SQL = "sql"
    LMDB = "lmdb"


class SupportRole(object):
    LOCAL = "local"
    GUEST = "guest"
    HOST = "host"
    ARBITER = "arbiter"

    @classmethod
    def support_roles(cls):
        return [cls.LOCAL, cls.GUEST, cls.HOST, cls.ARBITER]


