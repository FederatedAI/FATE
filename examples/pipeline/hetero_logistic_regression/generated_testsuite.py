import json
import os
import sys

cur_path = os.path.realpath(__file__)
for i in range(4):
    cur_path = os.path.dirname(cur_path)
print(f'fate_path: {cur_path}')
sys.path.append(cur_path)

cur_dir = os.path.abspath(os.path.dirname(__file__))


def insert_extract_code(file_path, fold_name):
    f_str = open(cur_dir + '/' + file_path, 'r').read()

    code = \
        """
    from examples.pipeline.{}.generated_testsuite import extract
    extract(pipeline, __file__)
        """.format(fold_name)

    f_str = f_str.replace('pipeline.fit(work_mode=work_mode)',
                          '# pipeline.fit(work_mode=work_mode)\n' + code)
    f_str = f_str.replace('common_tools.prettify(pipeline.get_component("hetero_lr_0").get_summary())',
                          '')
    f_str = f_str.replace('common_tools.prettify(pipeline.get_component("evaluation_0").get_summary())',
                          '')
    f_str = f_str.replace('for i in range(4):',
                          'for i in range(5):')
    return f_str


def extract(my_pipeline, file_name, output_path='generated_conf_and_dsl'):
    out_name = file_name.split('/')[-1]
    out_name = out_name.replace('pipeline-', '').replace('.py', '').replace('-', '_')
    conf = my_pipeline.get_train_conf()
    dsl = my_pipeline.get_train_dsl()
    conf_name = './{}/{}_conf.json'.format(output_path, out_name)
    dsl_name = './{}/{}_dsl.json'.format(output_path, out_name)
    json.dump(conf, open(conf_name, 'w'), indent=4)
    print('conf name is {}'.format(conf_name))
    json.dump(dsl, open(dsl_name, 'w'), indent=4)
    print('dsl name is {}'.format(dsl_name))


def get_testsuite_file(testsuite_file_path):
    import examples
    cpn_path = os.path.dirname(examples.__file__) + f'/dsl/v1/{testsuite_file_path}'
    with open(cpn_path, 'r', encoding='utf-8') as load_f:
        testsuite_json = json.load(load_f)
    testsuite_json['tasks'] = {}
    return testsuite_json


def do_generated(fold_name='hetero_logistic_regression'):
    folder = '.'
    files = os.listdir(".")
    cmd = 'python {}'

    replaced_path = 'replaced_code'
    generated_path = 'generated_conf_and_dsl'

    if not os.path.exists('./{}'.format(replaced_path)):
        os.system('mkdir {}'.format(replaced_path))

    if not os.path.exists('./{}'.format(generated_path)):
        os.system('mkdir {}'.format(generated_path))

    for f in files:
        if not f.startswith("pipeline"):
            continue
        print(f)
        code_str = insert_extract_code(f, fold_name)
        open('./{}/{}'.format(replaced_path, f), 'w').write(code_str)
        print('replace done')
        # file_path = folder + f
        # os.system(cmd.format(folder + f))

    exe_files = os.listdir('./{}/'.format(replaced_path))
    for f in exe_files:
        print('executing {}'.format(f))
        os.system(cmd.format('./{}/'.format(replaced_path) + f))

    suite_json = get_testsuite_file('hetero_logistic_regression/hetero_lr_testsuite.json')
    conf_files = os.listdir('./{}/'.format(generated_path))
    f_dsl = {"-".join(f.split('_')[2: -1]): f for f in conf_files if 'dsl.json' in f}
    f_conf = {"-".join(f.split('_')[2: -1]): f for f in conf_files if 'conf.json' in f}

    for task_type, dsl_file in f_dsl.items():
        conf_file = f_conf[task_type]
        suite_json['tasks'][task_type] = {
            "conf": conf_file,
            "dsl": dsl_file
        }

    with open('./{}/{}_testsuite.json'.format(generated_path, fold_name), 'w', encoding='utf-8') as json_file:
        json.dump(suite_json, json_file, ensure_ascii=False, indent=4)

    # os.system('rm -rf {}'.format(replaced_path))
from sklearn.metrics import fowlkes_mallows_score

if __name__ == '__main__':
    do_generated()
    # pass
