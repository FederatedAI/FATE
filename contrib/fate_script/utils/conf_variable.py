from arch.api.utils import file_utils

class ConfVar:
    def __init__(self):
        self.iter_num = 2
        self.batch_num = 1
        self.learning_rate = 0.15
        self.eps = 1e-4
	
    def init_conf(self, role):
        conf_path = file_utils.load_json_conf('contrib/fate_script/conf/' + str(role) + '_runtime_conf.json')
        self.iter_num = conf_path.get("FATEScriptLRParam").get("iter_num")
        self.batch_num = conf_path.get("FATEScriptLRParam").get("batch_num")
        self.learning_rate = conf_path.get("FATEScriptLRParam").get("learning_rate")
        self.eps = conf_path.get("FATEScriptLRParam").get("eps")
