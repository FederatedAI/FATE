import logging
import typing
from dataclasses import dataclass, field

from fate.arch.launchers.argparser import HfArgumentParser
from fate.arch.launchers.multiprocess_launcher import launch

if typing.TYPE_CHECKING:
    from fate.arch import Context

logger = logging.getLogger(__name__)


@dataclass
class SSHEArguments:
    lr: float = field(default=0.05)
    guest_data: str = field(default=None)
    host_data: str = field(default=None)


def run_sshe_lr(ctx: "Context"):
    from fate.ml.mpc.sshe_lr import SSHELogisticRegression
    from fate.arch import dataframe

    ctx.mpc.init()
    args, _ = HfArgumentParser(SSHEArguments).parse_args_into_dataclasses(return_remaining_strings=True)
    inst = SSHELogisticRegression(args.lr)
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


if __name__ == "__main__":
    launch(run_sshe_lr, extra_args_desc=[SSHEArguments])
