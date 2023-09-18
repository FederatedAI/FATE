import sys

arbiter = ("arbiter", 10000)
guest = ("guest", 10000)
host = ("host", 9999)
name = "fed"


def create_ctx(local):
    from fate.arch import Context
    from fate.arch.computing.standalone import CSession
    from fate.arch.federation.standalone import StandaloneFederation
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    computing = CSession()
    return Context(computing=computing,
                   federation=StandaloneFederation(computing, name, local, [guest, host, arbiter]))


if __name__ == "__main__":
    if sys.argv[1] == "guest":
        from fate.arch.protocol import SecureAggregatorClient
        import numpy as np

        ctx = create_ctx(guest)
        client = SecureAggregatorClient(is_mock=True)
        client.dh_exchange(ctx, [ctx.guest.rank, *ctx.hosts.ranks])
        print('ranks are {}'.format([ctx.guest.rank, *ctx.hosts.ranks]))
        print(client.secure_aggregate(ctx, [np.zeros((3, 4)), np.ones((2, 3))]))
    elif sys.argv[1] == "host":
        from fate.arch.protocol import SecureAggregatorClient
        import numpy as np

        ctx = create_ctx(host)
        client = SecureAggregatorClient(is_mock=True)
        client.dh_exchange(ctx, [ctx.guest.rank, *ctx.hosts.ranks])
        print(client.secure_aggregate(ctx, [np.zeros((3, 4)), np.ones((2, 3))]))
    else:
        from fate.arch.protocol import SecureAggregatorServer

        ctx = create_ctx(arbiter)
        server = SecureAggregatorServer([ctx.guest.rank, *ctx.hosts.ranks], is_mock=True)
        server.secure_aggregate(ctx)