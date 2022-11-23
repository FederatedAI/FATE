class JobConf(object):
    def __init__(self):
        self._conf = dict()

    def set(self, k, v):
        self._conf[k] = v

    def set_all(self, **kwargs):
        self._conf.update(kwargs)

    @property
    def conf(self):
        return self._conf
