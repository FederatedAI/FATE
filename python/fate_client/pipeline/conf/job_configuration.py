class JobConf(object):
    def __init__(self):
        self._conf = dict()

    def set(self, k, v):
        self._conf[k] = v

    def set_all(self, **kwargs):
        self._conf.update(kwargs)

    def update(self, conf: dict):
        for k, v in conf.items():
            if k not in self._conf:
                self._conf[k] = v

    def dict(self):
        return self._conf


class TaskConf(JobConf):
    ...
