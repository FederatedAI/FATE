import sys
import torch as t


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

    epoch = 10

    if sys.argv[1] == "guest":
        from fate.ml.aggregator.plaintext_aggregator import PlainTextAggregatorClient

        ctx = create_ctx(guest)
        client = PlainTextAggregatorClient(ctx, sample_num=100, aggregate_type='weighted_mean')
        model = t.nn.Sequential(
            t.nn.Linear(10, 10),
            t.nn.ReLU(),
            t.nn.Linear(10, 1),
            t.nn.Sigmoid()
        )

        for i in range(epoch):
            client.model_aggregation(model)
    elif sys.argv[1] == "host":
        from fate.ml.aggregator.plaintext_aggregator import PlainTextAggregatorClient

        ctx = create_ctx(host)
        client = PlainTextAggregatorClient(ctx, sample_num=100, aggregate_type='weighted_mean')
        model = t.nn.Sequential(
            t.nn.Linear(10, 10),
            t.nn.ReLU(),
            t.nn.Linear(10, 1),
            t.nn.Sigmoid()
        )

        for i in range(epoch):
            client.model_aggregation(model)

    else:

        from fate.ml.aggregator.plaintext_aggregator import PlainTextAggregatorServer
        ctx = create_ctx(arbiter)
        server = PlainTextAggregatorServer(ctx)

        for i in range(epoch):
            server.model_aggregation()

