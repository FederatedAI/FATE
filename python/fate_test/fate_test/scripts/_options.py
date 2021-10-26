import time

import click
from fate_test._config import parse_config, default_config
from fate_test.scripts._utils import _set_namespace


class SharedOptions(object):
    _options = {
        "config": (('-c', '--config'),
                   dict(type=click.Path(exists=True), help=f"Manual specify config file", default=None),
                   default_config().__str__()),
        "namespace": (('-n', '--namespace'),
                      dict(type=str, help=f"Manual specify fate_test namespace", default=None),
                      time.strftime('%Y%m%d%H%M%S')),
        "namespace_mangling": (('-nm', '--namespace-mangling',),
                               dict(type=bool, is_flag=True, help="Mangling data namespace", default=None),
                               False),
        "yes": (('-y', '--yes',), dict(type=bool, is_flag=True, help="Skip double check", default=None),
                False),
        "extend_sid": (('--extend_sid', ),
                       dict(type=bool, is_flag=True, help="whether to append uuid as sid when uploading data",
                            default=None), False),
        "auto_increasing_sid": (('--auto_increasing_sid', ),
                                dict(type=bool, is_flag=True, help="whether to generate sid value starting at 0",
                                     default=None), False),
    }

    def __init__(self):
        self._options_kwargs = {}

    def __getitem__(self, item):
        return self._options_kwargs[item]

    def get(self, k, default=None):
        v = self._options_kwargs.get(k, default)
        if v is None and k in self._options:
            v = self._options[k][2]
        return v

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                self._options_kwargs[k] = v

    def post_process(self):
        # add defaults here
        for k, v in self._options.items():
            if self._options_kwargs.get(k, None) is None:
                self._options_kwargs[k] = v[2]

        # update config
        config = parse_config(self._options_kwargs['config'])
        self._options_kwargs['config'] = config

        _set_namespace(self._options_kwargs['namespace_mangling'], self._options_kwargs['namespace'])

    @classmethod
    def get_shared_options(cls, hidden=False):
        def shared_options(f):
            for name, option in cls._options.items():
                f = click.option(*option[0], **dict(option[1], hidden=hidden))(f)
            return f

        return shared_options
