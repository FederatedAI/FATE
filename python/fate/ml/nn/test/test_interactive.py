from fate.ml.nn.hetero.agg_layer.plaintext_agg_layer import InteractiveLayerGuest, InteractiveLayerHost
import sys
from datetime import datetime


def get_current_datetime_str():
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


guest = ("guest", "10000")
host = ("host", "9999")
name = get_current_datetime_str()


def create_ctx(local, context_name):
    from fate.arch import Context
    from fate.arch.computing.standalone import CSession
    from fate.arch.federation.standalone import StandaloneFederation
    import logging

    # prepare log
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    # init fate context
    computing = CSession()
    return Context(
        computing=computing, federation=StandaloneFederation(computing, context_name, local, [guest, host])
    )


if __name__ == "__main__":

    party = sys.argv[1]
    import torch as t

    def set_seed(seed):
        t.manual_seed(seed)
        if t.cuda.is_available():
            t.cuda.manual_seed_all(seed)
            t.backends.cudnn.deterministic = True
            t.backends.cudnn.benchmark = False

    set_seed(42)

    if party == "guest":

        ctx = create_ctx(guest, get_current_datetime_str())
        layer = InteractiveLayerGuest(ctx, 8, 10, 10)
        fake_features = t.randn(2, 10)
        out = layer.forward(fake_features)

        fake_top_model = t.nn.Sequential(
            t.nn.Linear(8, 1),
            t.nn.Sigmoid()
        )

        loss = t.nn.BCELoss()

        print('guest model weight is {}'.format(layer._guest_model.weight))

        out.requires_grad = True
        top_out = fake_top_model(out)
        fake_label = t.randint(0, 2, (2, 1)).type(t.float32)
        loss_val = loss(top_out, fake_label)
        from torch.autograd import grad
        error = grad(loss_val, out)[0]
        ret = layer.backward(error)

    elif party == "host":

        ctx = create_ctx(host, get_current_datetime_str())
        layer = InteractiveLayerHost(ctx)
        fake_features = t.randn(2, 10)
        layer.forward(fake_features)
        error = layer.backward()
