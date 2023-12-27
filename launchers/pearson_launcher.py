import logging
import typing
from dataclasses import dataclass, field

from fate.arch.launchers.argparser import HfArgumentParser
from fate.arch.launchers.multiprocess_launcher import launch

if typing.TYPE_CHECKING:
    from fate.arch import Context

logger = logging.getLogger(__name__)


@dataclass
class PearsonArguments:
    guest_data: str = field()
    host_data: str = field()


def run_pearson(ctx: "Context"):
    from fate.ml.statistics.pearson_correlation import PearsonCorrelation
    from fate.arch import dataframe

    ctx.mpc.init()
    args, _ = HfArgumentParser(PearsonArguments).parse_args_into_dataclasses(return_remaining_strings=True)
    inst = PearsonCorrelation()
    if ctx.is_on_guest:
        kwargs = {
            "sample_id_name": None,
            "match_id_name": "id",
            "delimiter": ",",
            "label_name": "y",
            "label_type": "float32",
            "dtype": "float32",
        }
        input_data = dataframe.CSVReader(**kwargs).to_frame(ctx, args.guest_data)
    else:
        kwargs = {
            "sample_id_name": None,
            "match_id_name": "id",
            "delimiter": ",",
            "dtype": "float32",
        }
        input_data = dataframe.CSVReader(**kwargs).to_frame(ctx, args.host_data)
    inst.fit(ctx, input_data=input_data)
    print(f"role: {ctx.local.role};\n vif: {inst.vif}")

    # validate
    # if ctx.is_on_guest:
    #     import pandas as pd
    #
    #     x = pd.read_csv(args.guest_data).drop(columns=["id", "y"])
    #     y = pd.read_csv(args.host_data).drop(columns=["id"])
    #     x = (x - x.mean()) / x.std()
    #     y = (y - y.mean()) / y.std()
    #     n = x.shape[0]
    #     expect_corr = x.to_numpy().transpose() @ y.to_numpy()
    #     expect_corr /= n
    #     logger.info(f"expect_corr={corr}")
    #     logger.info(f"error={abs(corr.numpy() - expect_corr).max()}")


if __name__ == "__main__":
    launch(run_pearson, extra_args_desc=[PearsonArguments])
