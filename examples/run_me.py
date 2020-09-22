import os
from examples.extractor import insert_extract_code

folder = './pipeline/hetero_sbt/'
files = os.listdir(folder)
cmd = 'python {}'

replaced_path = 'replaced_code'
generated_path = 'generated_conf_and_dsl'

if not os.path.exists('./{}'.format(replaced_path)):
    os.system('mkdir {}'.format(replaced_path))

if not os.path.exists('./{}'.format(generated_path)):
    os.system('mkdir {}'.format(generated_path))

for f in files:
    print(f)
    code_str = insert_extract_code(folder + f)
    open('./{}/{}'.format(replaced_path, f), 'w').write(code_str)
    print('replace done')
    # file_path = folder + f
    # os.system(cmd.format(folder + f))

exe_files = os.listdir('./{}/'.format(replaced_path))
for f in exe_files:
    print('executing {}'.format(f))
    os.system(cmd.format('./{}/'.format(replaced_path) + f))

# os.system()