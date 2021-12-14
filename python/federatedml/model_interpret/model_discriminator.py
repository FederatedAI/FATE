from federatedml.util import consts
from federatedml.components.components import Components
"""
Given a model protobuf, return corresponding SHAP explainer, extracted meta and param
"""


def get_model_info(model_dict):
    """
    hetero or homo ?
    """

    # discriminate
    tree_prefix = consts.HETERO_SBT_GUEST_MODEL.replace('Guest', '').replace('Hetero', '')
    fed_type = None
    for key in model_dict:
        for model_key in model_dict[key]:

            if 'Hetero' in model_key:
                fed_type = consts.HETERO
            elif 'Homo' in model_key:
                fed_type = consts.HOMO

            if tree_prefix in model_key:
                if fed_type == consts.HETERO:
                    return consts.HETERO_SBT, fed_type
                elif fed_type == consts.HOMO:
                    return consts.HOMO_SBT, fed_type
                else:
                    raise ValueError('can not recognize this model')

    return None


def model_discriminator(model_dict):

    meta, param = None, None
    key = 'isometric_model'
    model_name, fed_type = get_model_info(model_dict[key])

    for model_key in model_dict[key]:
        model_content = model_dict[key][model_key]

        for content_name in model_content:

            if 'Meta' in content_name:
                meta = model_content[content_name]

            elif 'Param' in content_name:
                param = model_content[content_name]

    return model_name, meta, param


def extract_model(model_dict, role):

    meta, param = None, None
    key = 'isometric_model'

    for model_key in model_dict[key]:
        model_content = model_dict[key][model_key]

        for content_name in model_content:

            if 'Meta' in content_name:
                meta = model_content[content_name]

            elif 'Param' in content_name:
                param = model_content[content_name]

    module_name = meta.module
    cpn_meta = Components.get(module_name, None)
    algo_class = cpn_meta.get_run_obj(role)

    return meta, param, module_name, algo_class


def reconstruct_model_dict(model_dict):

    new_model_dict = {'model': model_dict['isometric_model']}
    return new_model_dict



