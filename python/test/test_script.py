import contextlib
import sys
from subprocess import Popen

from fate.arch import Backend, Context
from fate.arch.computing.standalone import CSession
from fate.arch.context import Context, disable_inner_logs
from fate.arch.federation.standalone import StandaloneFederation


def host(federation_id, party, parties):
    disable_inner_logs()
    computing = CSession()
    federation = StandaloneFederation(computing, federation_id, party, parties)
    ctx = Context(
        "guest", backend=Backend.STANDALONE, computing=computing, federation=federation
    )
    ctx.cipher.phe.keygen()
    with ctx.sub_ctx("predict") as sub_ctx:
        sub_ctx.log.debug("ctx inited")
        loss = 0.2
        sub_ctx.guest.push("loss", loss)
        guest_loss = sub_ctx.guest.pull("loss").unwrap()
        sub_ctx.summary.add("guest_loss", guest_loss)
        ctx.log.debug(f"{sub_ctx.summary.summary}")
    print(ctx.tensor.random_tensor((10, 10)))


def guest(federation_id, party, parties):
    disable_inner_logs()
    computing = CSession()
    federation = StandaloneFederation(computing, federation_id, party, parties)
    ctx = Context("host", computing=computing, federation=federation)
    with ctx.sub_ctx("predict") as sub_ctx:
        sub_ctx.log.error("ctx inited")
        loss = 0.1
        sub_ctx.hosts.push("loss", loss)
        host_loss = sub_ctx.hosts(0).pull("loss").unwrap()
        sub_ctx.summary.add("host_loss", host_loss)
        ctx.log.debug(f"{sub_ctx.summary.summary}")


if __name__ == "__main__":
    import argparse
    import json
    import tempfile

    parser = argparse.ArgumentParser()
    parser.add_argument("--role", default=None)
    parser.add_argument("--path", default=None)
    args = parser.parse_args()
    if not args.role:
        federation_id = "federation_id"
        guest_party = ("guest", "guest_party_id")
        host_party = ("host", "host_party_id")
        parties = [guest_party, host_party]
        with contextlib.ExitStack() as stack:
            f = stack.enter_context(tempfile.NamedTemporaryFile(mode="w"))
            json.dump(
                dict(party=guest_party, parties=parties, federation_id=federation_id), f
            )
            f.flush()
            p1 = Popen([sys.executable, __file__, "--role", "guest", "--path", f.name])
            f = stack.enter_context(tempfile.NamedTemporaryFile(mode="w"))
            json.dump(
                dict(party=host_party, parties=parties, federation_id=federation_id), f
            )
            f.flush()
            p2 = Popen([sys.executable, __file__, "--role", "host", "--path", f.name])
            p1.communicate()
            p2.communicate()
    elif args.role == "host":
        with open(args.path) as f:
            config = json.load(f)
            config["party"] = tuple(config["party"])
            config["parties"] = [tuple(p) for p in config["parties"]]
            host(**config)
    elif args.role == "guest":
        with open(args.path) as f:
            config = json.load(f)
            config["party"] = tuple(config["party"])
            config["parties"] = [tuple(p) for p in config["parties"]]
            guest(**config)
