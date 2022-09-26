from multiprocessing import Process

from fate.arch.computing.standalone import CSession
from fate.arch.context import Context
from fate.arch.federation.standalone import StandaloneFederation


def host(federation_id, local_party, parties):
    computing = CSession()
    federation = StandaloneFederation(computing, federation_id, local_party, parties)
    ctx = Context("guest", computing=computing, federation=federation)
    with ctx.sub_ctx("predict") as sub_ctx:
        sub_ctx.log.debug("ctx inited")
        loss = 0.2
        sub_ctx.guest.push("loss", loss)
        guest_loss = sub_ctx.guest.pull("loss").unwrap()
        sub_ctx.summary.add("guest_loss", guest_loss)
        ctx.log.debug(f"{sub_ctx.summary.summary}")


def guest(federation_id, local_party, parties):
    computing = CSession()
    federation = StandaloneFederation(computing, federation_id, local_party, parties)
    ctx = Context("host", computing=computing, federation=federation)
    with ctx.sub_ctx("predict") as sub_ctx:
        sub_ctx.log.error("ctx inited")
        loss = 0.1
        sub_ctx.hosts.push("loss", loss)
        host_loss = sub_ctx.hosts(0).pull("loss").unwrap()
        sub_ctx.summary.add("host_loss", host_loss)
        ctx.log.debug(f"{sub_ctx.summary.summary}")


if __name__ == "__main__":

    federation_id = "federation_id"
    guest_party = ("guest", "guest_party_id")
    host_party = ("host", "host_party_id")
    parties = [guest_party, host_party]
    p_guest = Process(target=guest, args=(federation_id, guest_party, parties))
    p_host = Process(target=host, args=(federation_id, host_party, parties))
    p_guest.start()
    p_host.start()
    p_guest.join()
    p_host.join()
