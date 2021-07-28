from fate_arch.common.conf_utils import get_base_config


def override_base_config(config):
    def _get_base_config(key, *args, **kwargs):
        for k, v in config.items():
            if k == key:
                return v
        return get_base_config(key, *args, **kwargs)
    return _get_base_config
