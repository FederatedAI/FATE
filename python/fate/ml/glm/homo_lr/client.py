from typing import Optional
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.ml.abc.module import HomoModule
from fate.arch import Context
import logging
import pandas as pd


logger = logging.getLogger(__name__)


class Data(object):

    def __init__(self, features: pd.DataFrame, sample_ids: pd.DataFrame, match_ids: pd.DataFrame, labels: pd.DataFrame) -> None:
        # set var
        self.features = features
        self.sample_ids = sample_ids
        self.match_ids = match_ids
        self.labels = labels

    @staticmethod
    def from_fate_dataframe(df: DataFrame):
        schema = df.schema
        sample_id = schema.sample_id_name
        match_id = schema.match_id_name
        label = schema.label_name
        pd_df = df.as_pd_df()
        features = pd_df.drop([sample_id, match_id, label], axis=1)
        sample_ids = pd_df[[sample_id]]
        match_ids = pd_df[[match_id]]
        labels = pd_df[[label]]
        return Data(features, sample_ids, match_ids, labels)


class HomoLRClient(HomoModule):

    def __init__(self) -> None:
        super().__init__()
        self.df_schema = None
        self.train_data = None
        self.validate_data = None
        self.predict_data = None

    def fit(self, ctx: Context, train_data: DataFrame, validate_data: DataFrame = None) -> None:

        train_pd_df = DataFrame.as_pd_df()
        if validate_data is not None:
            validate_pd_df = DataFrame.as_pd_df()

        
    
    def predict(self, ctx: Context, predict_data: DataFrame) -> DataFrame:
        return super().predict(ctx, predict_data)