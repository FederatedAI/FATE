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


def run_sshe_linr(ctx: "Context"):
    from fate.ml.glm.hetero.sshe import SSHELinearRegression
    from fate.arch import dataframe

    ctx.mpc.init()
    args, _ = HfArgumentParser(SSHEArguments).parse_args_into_dataclasses(return_remaining_strings=True)
    inst = SSHELinearRegression(
        epochs=5,
        batch_size=300,
        tol=0.01,
        early_stop="diff",
        learning_rate=0.15,
        init_param={"method": "random_uniform", "fit_intercept": True, "random_state": 1},
        reveal_every_epoch=False,
        reveal_loss_freq=2,
        threshold=0.5,
    )
    if ctx.is_on_guest:
        kwargs = {
            "sample_id_name": None,
            "match_id_name": "idx",
            "delimiter": ",",
            "label_name": "motor_speed",
            "label_type": "float32",
            "dtype": "float32",
        }
        input_data = dataframe.CSVReader(**kwargs).to_frame(ctx, args.guest_data)
    else:
        kwargs = {
            "sample_id_name": None,
            "match_id_name": "idx",
            "delimiter": ",",
            "dtype": "float32",
        }
        input_data = dataframe.CSVReader(**kwargs).to_frame(ctx, args.host_data)
    inst.fit(ctx, train_data=input_data)
    print(f"model: {inst.get_model()}")


if __name__ == "__main__":
    launch(run_sshe_linr, extra_args_desc=[SSHEArguments])
