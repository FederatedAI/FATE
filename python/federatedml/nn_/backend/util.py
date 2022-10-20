ML_PATH = 'federatedml.nn_'


def paraent_attr_check(obj, attr):
    if not hasattr(obj, attr):
        raise ValueError('{} is None, please make sure that you call the __init__ function of the super class')
        
