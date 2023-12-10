import os


def get_base_dir():
    return os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, os.pardir, os.pardir, os.pardir)
    )
