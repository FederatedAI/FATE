import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

class Config(object):
    def __init__(self, config_file):
        with open(config_file, "r") as f:
            conf = yaml.load(f, Loader=Loader)
            parties = conf.get("parties", {})
            if len(parties) == 0:
                raise ValueError(f"Parties id must be specified.")
            self.host = parties["host"]
            self.guest = parties["guest"][0]
            self.arbiter = parties["arbiter"][0]
            self.backend = conf.get("backend", 0)
            self.work_mode = conf.get("work_mode", 0)
