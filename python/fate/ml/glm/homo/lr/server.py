from fate.ml.abc.module import HomoModule
from fate.arch import Context
from fate.arch.dataframe import DataFrame
from fate.arch import Context
import logging
from fate.ml.nn.algo.homo.fedavg import FedAVGServer


logger = logging.getLogger(__name__)


class HomoLRServer(HomoModule):

    def __init__(self) -> None:
        pass

    def fit(self, ctx: Context, data: DataFrame = None) -> None:

        server = FedAVGServer(ctx=ctx)
        logger.info('server class init done, start fed training')
        server.train()
        logger.info('homo lr fit done')

    def predict(
            self,
            ctx: Context,
            predict_data: DataFrame = None) -> DataFrame:

        logger.info('kkip prediction stage')
