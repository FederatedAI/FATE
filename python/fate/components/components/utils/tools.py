from fate.arch.dataframe import DataFrame
from .consts import TRAIN_SET, VALIDATE_SET, TESET_SET


def cat_train_and_validate_df(train_df: DataFrame, val_df: DataFrame):
    """
    Concatenate train and validate dataframe
    """
    return train_df.vstack(val_df)


def add_dataset_type(df: DataFrame, dataset_type):
    assert dataset_type in [TRAIN_SET, VALIDATE_SET, TESET_SET], f"dataset_type must be one of {TRAIN_SET}, {VALIDATE_SET}, {TESET_SET}"
    return df


