import logging
import typing
from dataclasses import dataclass, field

from fate.arch.launchers.argparser import HfArgumentParser
from fate.arch.launchers.multiprocess_launcher import launch

if typing.TYPE_CHECKING:
    from fate.arch import Context

logger = logging.getLogger(__name__)


@dataclass
class PSIArguments:
    guest_data: str = field(default=None)
    host_data: str = field(default=None)


def run_psi(ctx: "Context"):
    from fate.arch import dataframe
    from fate.arch.protocol.psi import psi_run

    args, _ = HfArgumentParser(PSIArguments).parse_args_into_dataclasses(return_remaining_strings=True)
    if ctx.is_on_guest:
        kwargs = {
            "sample_id_name": None,
            "match_id_name": "id",
            "delimiter": ",",
            "label_name": "y",
            "label_type": "int32",
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

    psi_run(ctx, df=input_data)


if __name__ == "__main__":
    launch(run_psi, extra_args_desc=[PSIArguments])
