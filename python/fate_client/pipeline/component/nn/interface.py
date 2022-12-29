from pipeline.param.base_param import BaseParam
import sys


def not_working_save_to_fate(*args, **kwargs):
    raise ValueError(
        'save to fate not working, please check if your ipython is installed, '
        'and if ipython.get_ipython() is working')


try:
    import IPython as ipy
    from IPython.core.magic import register_cell_magic
except ImportError as e:
    ipy = None
    register_cell_magic = None


# check
if register_cell_magic is not None:
    if ipy.get_ipython():
        @register_cell_magic
        def save_to_fate(line, cell):

            # search for federatedml path
            base_path = None
            for p in sys.path:
                if p.endswith('/fate/python'):
                    base_path = p
                    break

            if base_path is None:
                raise ValueError(
                    'cannot find fate/python in system path, please check your configuration')

            base_path = base_path + '/federatedml/'

            model_pth = 'nn/model_zoo/'
            dataset_pth = 'nn/dataset/'
            trainer_pth = 'nn/homo/trainer/'
            aggregator_pth = 'framework/homo/aggregator/'
            loss_path = 'nn/loss/'

            mode_map = {
                'model': model_pth,
                'trainer': trainer_pth,
                'aggregator': aggregator_pth,
                'dataset': dataset_pth,
                'loss': loss_path
            }

            args = line.split()
            assert len(
                args) == 2, "input args len is not 2, got {} \n expect format: %%save_to_fate SAVE_MODE FILENAME \n SAVE_MODE in ['model', 'dataset', 'trainer', 'loss', 'aggregator']   FILE_NAME xxx.py".format(args)
            modes_avail = ['model', 'dataset', 'trainer', 'aggregator', 'loss']
            save_mode = args[0]
            file_name = args[1]
            assert save_mode in modes_avail, 'avail modes are {}, got {}'.format(
                modes_avail, save_mode)
            assert file_name.endswith('.py'), 'save file should be a .py'
            with open(base_path + mode_map[save_mode] + file_name, 'w') as f:
                f.write(cell)
            ipy.get_ipython().run_cell(cell)
    else:
        save_to_fate = not_working_save_to_fate
else:
    save_to_fate = not_working_save_to_fate


class TrainerParam(BaseParam):

    def __init__(self, trainer_name=None, **kwargs):
        super(TrainerParam, self).__init__()
        self.trainer_name = trainer_name
        self.param = kwargs

    def check(self):
        if self.trainer_name is None:
            raise ValueError(
                'You did not specify the trainer name, please set the trainer name')
        self.check_string(self.trainer_name, 'trainer_name')

    def to_dict(self):
        ret = {'trainer_name': self.trainer_name, 'param': self.param}
        return ret


class DatasetParam(BaseParam):

    def __init__(self, dataset_name=None, **kwargs):
        super(DatasetParam, self).__init__()
        self.dataset_name = dataset_name
        self.param = kwargs

    def check(self):
        self.check_string(self.dataset_name, 'dataset_name')

    def to_dict(self):
        ret = {'dataset_name': self.dataset_name, 'param': self.param}
        return ret
