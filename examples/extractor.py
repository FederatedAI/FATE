import json


def insert_extract_code(file_path):

    f_str = open(file_path, 'r').read()

    code = \
    """
    from examples.extractor import extract
    extract(pipeline, __file__)
    """

    f_str = f_str.replace('pipeline.fit(backend=backend, work_mode=work_mode)',
                  '# pipeline.fit(backend=backend, work_mode=work_mode)\n' + code)

    return f_str


def extract(pipeline, file_name, output_path='generated_conf_and_dsl'):
    out_name = file_name.split('/')[-1]
    out_name = out_name.replace('pipeline', '').replace('.py', '')
    conf = pipeline.get_train_conf()
    dsl = pipeline.get_train_dsl()
    conf_name = './{}/{}_conf.json'.format(output_path, out_name)
    dsl_name = './{}/{}_dsl.json'.format(output_path, out_name)
    json.dump(conf, open(conf_name, 'w'), indent=4)
    print('conf name is {}'.format(conf_name))
    json.dump(dsl, open(dsl_name, 'w'), indent=4)
    print('dsl name is {}'.format(dsl_name))
